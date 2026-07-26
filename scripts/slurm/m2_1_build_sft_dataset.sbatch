#!/bin/bash
#SBATCH --job-name=m2_1_sft_data
#SBATCH --partition=compute
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --output=logs/m2_1_sft_data_%j.out
#SBATCH --error=logs/m2_1_sft_data_%j.err

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

echo "=== M2.1 SFT Dataset Construction ==="
echo "Job: ${SLURM_JOB_ID:-none}, Seed: $SEED"
echo "Started: $(date)"

# Step 0: Tests
echo "=== Step 0: Tests ==="
python -m pytest -q 2>&1 | tail -5

# Step 1: M2.0 Metrics Audit
echo "=== Step 1: M2.0 Metrics Audit ==="
python -m miniwebwork.audit.m2_0_metrics_audit artifacts/m2_0 artifacts/m2_0/m2_0_metrics_audit.json

# Step 2: Freeze test set
echo "=== Step 2: Freeze test set ==="
python -c "
import json, hashlib
from pathlib import Path
pub = Path('data/tasks/tasks_public.jsonl').read_text()
ora = Path('data/tasks/tasks_oracle.jsonl').read_text()
frozen = {'test_frozen': [], 'frozen_at': '2026-07-25', 'source_stage': 'M2.0'}
for line in pub.strip().split('\n'):
    if line.strip():
        t = json.loads(line)
        frozen['test_frozen'].append({'task_id': t['task_id'], 'task_type': t['task_type']})
frozen['public_sha256'] = hashlib.sha256(pub.encode()).hexdigest()
frozen['oracle_sha256'] = hashlib.sha256(ora.encode()).hexdigest()
Path('data/splits/m2_1/test_frozen.json').write_text(json.dumps(frozen, indent=2, ensure_ascii=False))
manifest = {'frozen_test_count': len(frozen['test_frozen']), 'public_sha256': frozen['public_sha256']}
Path('data/splits/m2_1/manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f'Frozen {len(frozen[\"test_frozen\"])} test tasks')
"

# Step 3: Generate tasks
echo "=== Step 3: Generate tasks ==="
python -m miniwebwork.data_generation.task_generator

# Step 4: Run expert trajectories
echo "=== Step 4: Expert trajectories ==="
python - <<'PYEOF'
import json, sys, time, uuid
from pathlib import Path
sys.path.insert(0, "src")
from miniwebwork.agent_env.environment import ProcurementBrowserEnv
from miniwebwork.data_generation.expert_agent import OracleExpertProcurementAgent

run_id = f"m2_1_expert_{uuid.uuid4().hex[:8]}"
results = {"train": [], "valid": []}

for split in ["train", "valid"]:
    tasks_path = Path(f"data/tasks/m2_1/{split}_oracle.jsonl")
    tasks = [json.loads(l) for l in tasks_path.read_text().strip().split("\n") if l.strip()]
    print(f"\n=== {split}: {len(tasks)} tasks ===")

    with ProcurementBrowserEnv(max_steps=20, run_id=run_id) as env:
        for i, oracle in enumerate(tasks):
            tid = oracle["task_id"]
            try:
                obs = env.reset(tid)
                agent = OracleExpertProcurementAgent(oracle, max_steps=20)
                agent.reset()
                env.set_agent_name("oracle_expert")
                done = False
                turns = []
                for step in range(20):
                    action = agent.act(obs)
                    result = env.step(action)
                    turns.append({"step": step, "action": action.to_dict(),
                                  "observation": obs.to_dict(),
                                  "result": {"reward": result.reward, "terminated": result.terminated,
                                             "truncated": result.truncated,
                                             "info": result.info}})
                    if result.observation:
                        obs = result.observation
                    if result.terminated or result.truncated:
                        done = True
                        break
                success = result.reward > 0
                traj = env.trajectory
                verdict = traj.verification if traj else {}
                traj_data = {"task_id": tid, "success": success, "steps": len(turns),
                             "reward": result.reward, "verification": verdict,
                             "termination_reason": result.info.get("termination_reason", ""),
                             "turns": turns}
                results[split].append(traj_data)
                status = "PASS" if success else "FAIL"
                print(f"  [{i+1}/{len(tasks)}] {tid}: {status} ({len(turns)} steps)")
            except Exception as e:
                print(f"  [{i+1}/{len(tasks)}] {tid}: ERROR {e}")
                results[split].append({"task_id": tid, "success": False, "error": str(e)[:200]})

# Save trajectories
for split in ["train", "valid"]:
    out = Path(f"data/expert/m2_1/{split}_trajectories.json")
    out.write_text(json.dumps(results[split], indent=2, ensure_ascii=False, default=str))

metrics = {
    "train": {"total": len(results["train"]), "success": sum(1 for r in results["train"] if r.get("success")),
              "rate": sum(1 for r in results["train"] if r.get("success")) / max(len(results["train"]), 1)},
    "valid": {"total": len(results["valid"]), "success": sum(1 for r in results["valid"] if r.get("success")),
              "rate": sum(1 for r in results["valid"] if r.get("success")) / max(len(results["valid"]), 1)},
}
Path("artifacts/m2_1/expert_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print(f"\nTrain: {metrics['train']['success']}/{metrics['train']['total']} ({metrics['train']['rate']:.1%})")
print(f"Valid: {metrics['valid']['success']}/{metrics['valid']['total']} ({metrics['valid']['rate']:.1%})")
PYEOF

# Step 5: Build SFT data
echo "=== Step 5: Build SFT data ==="
python - <<'PYEOF'
import json, hashlib, sys
from pathlib import Path
sys.path.insert(0, "src")
from miniwebwork.model_agent.prompt_builder import build_messages

prompt_sha = hashlib.sha256(Path("prompts/browser_agent_v1.txt").read_text().encode()).hexdigest()

for split in ["train", "valid"]:
    traj_path = Path(f"data/expert/m2_1/{split}_trajectories.json")
    trajectories = json.loads(traj_path.read_text())
    samples = []
    sample_idx = 0

    for traj in trajectories:
        if not traj.get("success"):
            continue
        turns = traj.get("turns", [])
        for turn in turns:
            obs_data = turn.get("observation", {})
            if not obs_data:
                continue
            # Reconstruct observation object
            from miniwebwork.agent_env.schemas import Observation, ElementDescriptor
            els = []
            for ed in obs_data.get("elements", []):
                els.append(ElementDescriptor(
                    element_id=ed.get("element_id",""), role=ed.get("role",""),
                    tag=ed.get("tag",""), name=ed.get("name",""), text=ed.get("text",""),
                    value=ed.get("value",""), input_type=ed.get("input_type",""),
                    testid=ed.get("testid",""), options=ed.get("options",[]),
                    disabled=ed.get("disabled", False)))
            obs = Observation(
                task_id=obs_data.get("task_id",""), episode_id=obs_data.get("episode_id",""),
                instruction=obs_data.get("instruction",""), step_index=obs_data.get("step_index",0),
                url=obs_data.get("url",""), path=obs_data.get("path",""),
                page_type=obs_data.get("page_type",""), title=obs_data.get("title",""),
                visible_text=obs_data.get("visible_text",""),
                text_truncated=obs_data.get("text_truncated", False),
                elements=els, last_action_result=obs_data.get("last_action_result"),
                terminal=obs_data.get("terminal", False))

            # Build messages using M2.0 PromptBuilder
            messages = build_messages(obs, [], version="browser_agent_v1")
            action = turn.get("action", {})
            assistant_content = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
            messages.append({"role": "assistant", "content": assistant_content})

            sample_idx += 1
            samples.append({
                "sample_id": f"{split[:1].upper()}{sample_idx:06d}",
                "split": split, "task_id": traj["task_id"],
                "task_type": traj.get("task_type", ""),
                "trajectory_id": traj.get("episode_id", traj.get("task_id", "")),
                "turn_index": turn.get("step", 0),
                "source": "oracle_expert",
                "prompt_version": "browser_agent_v1",
                "messages": messages,
                "action": action,
                "prompt_sha256": prompt_sha,
            })

    out_path = Path(f"data/sft/m2_1/{split}.jsonl")
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"{split}: {len(samples)} SFT samples")

# Manifest + stats
sft_manifest = {
    "schema_version": "1.0", "prompt_version": "browser_agent_v1",
    "train_samples": sum(1 for _ in open("data/sft/m2_1/train.jsonl")),
    "valid_samples": sum(1 for _ in open("data/sft/m2_1/valid.jsonl")),
}
Path("data/sft/m2_1/manifest.json").write_text(json.dumps(sft_manifest, indent=2, ensure_ascii=False))
print(json.dumps(sft_manifest))
PYEOF

# Step 6: Leakage audit
echo "=== Step 6: Leakage audit ==="
python - <<'PYEOF'
import json, sys
from pathlib import Path
frozen_ids = {t["task_id"] for t in json.loads(Path("data/splits/m2_1/test_frozen.json").read_text())["test_frozen"]}
leaks = {"oracle_leak": 0, "expected_product_leak": 0, "test_task_leak": 0, "cross_split_duplicate": 0}
for split in ["train", "valid"]:
    sft_path = Path(f"data/sft/m2_1/{split}.jsonl")
    if not sft_path.exists():
        continue
    for line in sft_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        s = json.loads(line)
        tid = s.get("task_id", "")
        if tid in frozen_ids:
            leaks["test_task_leak"] += 1
        # Check messages for oracle leaks
        for msg in s.get("messages", []):
            content = str(msg.get("content", ""))
            if "expected_product_id" in content:
                leaks["oracle_leak"] += 1
            # Check assistant for expected_product_id
            if msg["role"] == "assistant":
                try:
                    a = json.loads(content) if isinstance(content, str) else content
                    if "expected_product_id" in str(a):
                        leaks["expected_product_leak"] += 1
                except:
                    pass
# Cross-split instruction duplicate
train_insts = set()
for line in Path("data/tasks/m2_1/train_public.jsonl").read_text().strip().split("\n"):
    if line.strip():
        train_insts.add(json.loads(line)["instruction"])
for line in Path("data/tasks/m2_1/valid_public.jsonl").read_text().strip().split("\n"):
    if line.strip() and json.loads(line)["instruction"] in train_insts:
        leaks["cross_split_duplicate"] += 1
print(json.dumps(leaks, indent=2))
Path("artifacts/m2_1/m2_1_leakage_audit.json").write_text(json.dumps(leaks, indent=2, ensure_ascii=False))
PYEOF

# Step 7: Data quality stats
echo "=== Step 7: Data quality ==="
python - <<'PYEOF'
import json
from pathlib import Path
stats = {"train_samples": 0, "valid_samples": 0, "action_dist": {}, "task_type_dist": {}}
for split in ["train", "valid"]:
    path = Path(f"data/sft/m2_1/{split}.jsonl")
    if not path.exists():
        continue
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        s = json.loads(line)
        stats[f"{split}_samples"] = stats.get(f"{split}_samples", 0) + 1
        act = s.get("action", {}).get("action", "unknown")
        stats["action_dist"][act] = stats["action_dist"].get(act, 0) + 1
        tt = s.get("task_type", "unknown")
        stats["task_type_dist"][tt] = stats["task_type_dist"].get(tt, 0) + 1
Path("artifacts/m2_1/m2_1_data_quality.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
print(json.dumps(stats, indent=2, ensure_ascii=False))
PYEOF

echo "=== Complete at $(date) ==="
echo "=== Artifacts ==="
find artifacts/m2_1 -type f | sort
echo "Data files:"
find data/splits/m2_1 data/tasks/m2_1 data/sft/m2_1 data/expert/m2_1 -type f | sort
