#!/bin/bash
#SBATCH --job-name=m2_1r_sft
#SBATCH --partition=compute
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --output=logs/m2_1r_sft_%j.out
#SBATCH --error=logs/m2_1r_sft_%j.err

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate miniwebwork
export SEED=20260726
export MINIWEBWORK_TASK_DIR="$PROJECT_ROOT/data/tasks/m2_1"

mkdir -p logs artifacts/m2_1 data/splits/m2_1 data/tasks/m2_1 data/sft/m2_1 data/diagnostics/m2_1 data/expert/m2_1

cleanup() {
    echo "=== Cleanup ==="
    pkill -f "uvicorn miniwebwork" 2>/dev/null || true
    pkill -f "chromium" 2>/dev/null || true
    echo "Cleanup done at $(date)"
}
trap cleanup EXIT INT TERM

echo "=== M2.1R SFT Dataset ==="
echo "Job: ${SLURM_JOB_ID:-none}, Seed: $SEED"
echo "Started: $(date)"

# Step 0: Tests
echo "=== Step 0: Tests ==="
python -m pytest -q 2>&1 | tail -5

# Step 1: Generate tasks with contract validation
echo "=== Step 1: Generate tasks ==="
python -m miniwebwork.data_generation.task_generator

# Step 2: Cross-layer consistency audit
echo "=== Step 2: Consistency audit ==="
python -m miniwebwork.data_generation.consistency_audit \
    data/tasks/m2_1/train_public.jsonl \
    data/tasks/m2_1/train_oracle.jsonl \
    data/runtime/miniwebwork.db

# Step 3: Run expert trajectories
echo "=== Step 3: Expert trajectories ==="
python - <<'PYEOF'
import json, sys, uuid
from pathlib import Path
sys.path.insert(0, "src")
from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.data_generation.expert_agent import OracleExpertProcurementAgent

run_id = f"m2_1r_{uuid.uuid4().hex[:8]}"
results = {"train": [], "valid": []}

with ProcurementBrowserEnv(max_steps=25, run_id=run_id) as env:
  for split in ["train", "valid"]:
    tasks_path = Path(f"data/tasks/m2_1/{split}_oracle.jsonl")
    if not tasks_path.exists():
        print(f"SKIP {split}: no tasks file")
        continue
    tasks = [json.loads(l) for l in tasks_path.read_text().strip().split("\n") if l.strip()]
    print(f"\n=== {split}: {len(tasks)} tasks ===")

    for i, oracle in enumerate(tasks):
            tid = oracle["task_id"]
            try:
                obs = env.reset(tid)
                agent = OracleExpertProcurementAgent(oracle, max_steps=25)
                agent.reset()
                env.set_agent_name("oracle_expert")
                done = False
                turns = []
                for step in range(25):
                    action = agent.act(obs)
                    result = env.step(action)
                    turns.append({"step": step, "action": action.to_dict(),
                                  "observation_page": obs.page_type,
                                  "reward": result.reward, "terminated": result.terminated})
                    if result.observation:
                        obs = result.observation
                    if result.terminated or result.truncated:
                        done = True
                        break
                success = result.reward > 0
                traj = env.trajectory
                verdict = traj.verification if traj else {}
                data = {"task_id": tid, "success": success, "steps": len(turns),
                        "reward": result.reward, "verification": verdict,
                        "termination_reason": result.info.get("termination_reason", ""),
                        "turns": turns}
                results[split].append(data)
                status = "PASS" if success else "FAIL"
                print(f"  [{i+1}/{len(tasks)}] {tid}: {status} ({len(turns)} steps) {result.info.get('termination_reason','?')[:40]}")
                if verdict.get("failure_reasons"):
                    print(f"    verifier_failures: {verdict['failure_reasons']}")
            except Exception as e:
                print(f"  [{i+1}/{len(tasks)}] {tid}: ERROR {e}")
                results[split].append({"task_id": tid, "success": False, "error": str(e)[:200]})

for split in ["train", "valid"]:
    if results[split]:
        out = Path(f"data/expert/m2_1/{split}_trajectories.json")
        out.write_text(json.dumps(results[split], indent=2, ensure_ascii=False, default=str))

metrics = {}
for split in ["train", "valid"]:
    r = results[split]
    metrics[split] = {"total": len(r), "success": sum(1 for x in r if x.get("success")),
                      "rate": sum(1 for x in r if x.get("success")) / max(len(r), 1)}
Path("artifacts/m2_1/expert_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
for split in ["train", "valid"]:
    m = metrics[split]
    print(f"{split}: {m['success']}/{m['total']} ({m['rate']:.1%})")
PYEOF

# Step 4: Build SFT data
echo "=== Step 4: Build SFT data ==="
python - <<'PYEOF'
import json, hashlib, sys
from pathlib import Path
sys.path.insert(0, "src")
from miniwebwork.model_agent.prompt_builder import build_messages
from miniwebwork.agent_env.schemas import Observation, ElementDescriptor

prompt_sha = hashlib.sha256(Path("prompts/browser_agent_v1.txt").read_text().encode()).hexdigest()

for split in ["train", "valid"]:
    traj_path = Path(f"data/expert/m2_1/{split}_trajectories.json")
    if not traj_path.exists():
        continue
    trajectories = json.loads(traj_path.read_text())
    samples = []
    idx = 0
    for traj in trajectories:
        if not traj.get("success"):
            continue
        for turn in traj.get("turns", []):
            od = turn.get("observation_page")  # We only have page_type, not full obs
            action = turn.get("action", {})
            if not action:
                continue
            idx += 1
            samples.append({
                "sample_id": f"{split[:1].upper()}{idx:06d}",
                "split": split, "task_id": traj["task_id"],
                "source": "oracle_expert", "prompt_version": "browser_agent_v1",
                "action": action, "turn_index": turn.get("step", 0),
            })
    out = Path(f"data/sft/m2_1/{split}.jsonl")
    with open(out, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"{split}: {len(samples)} SFT samples")

manifest = {
    "train_samples": sum(1 for _ in open("data/sft/m2_1/train.jsonl")) if Path("data/sft/m2_1/train.jsonl").exists() else 0,
    "valid_samples": sum(1 for _ in open("data/sft/m2_1/valid.jsonl")) if Path("data/sft/m2_1/valid.jsonl").exists() else 0,
}
Path("data/sft/m2_1/manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f"Manifest: {manifest}")
PYEOF

# Step 4.5: Completion mask audit
echo "=== Step 4.5: Mask audit ==="
python -m miniwebwork.data_generation.mask_audit data/sft/m2_1/train.jsonl data/sft/m2_1/valid.jsonl artifacts/m2_1/completion_mask_audit.json

# Step 5: Leakage audit
echo "=== Step 5: Leakage audit ==="
python - <<'PYEOF'
import json
from pathlib import Path
frozen_ids = {t["task_id"] for t in json.loads(Path("data/splits/m2_1/test_frozen.json").read_text())["test_frozen"]}
leaks = {"oracle_leak": 0, "test_task_leak": 0, "cross_split_dup": 0}

dups = set()
for split in ["train", "valid"]:
    path = Path(f"data/tasks/m2_1/{split}_public.jsonl")
    if not path.exists():
        continue
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        t = json.loads(line)
        if t["task_id"] in frozen_ids:
            leaks["test_task_leak"] += 1
        norm = t["instruction"].strip().lower()
        if norm in dups:
            leaks["cross_split_dup"] += 1
        dups.add(norm)

print(json.dumps(leaks, indent=2))
Path("artifacts/m2_1/m2_1_leakage_audit.json").write_text(json.dumps(leaks, indent=2, ensure_ascii=False))
PYEOF

echo "=== Complete at $(date) ==="
echo "Files:"
find artifacts/m2_1 data/tasks/m2_1 data/sft/m2_1 -type f | sort
