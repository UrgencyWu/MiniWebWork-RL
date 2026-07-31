"""Machine-checkable M4 study configuration and provenance manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .data_generation.m4_rlvr import (
    DATASET_ID,
    DEFAULT_OUTPUT_DIR as DEFAULT_TASK_ROOT,
    DEFAULT_SEED_DIR,
    SPLIT_WORLD_COUNTS,
    assert_m4_split_purpose,
    validate_m4_rlvr_dataset,
)
from .m4_algorithms import AlgorithmSpec, get_algorithm_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "m4_rlvr_study_v1"
STUDY_SEEDS = (20260801, 20260802, 20260803)
ONLINE_GROUP_SIZE = 4
ONLINE_PASSES = 2
COLLECTED_ACTION_TOKEN_CAP = 250_000
EVAL_ROLLOUTS_PER_TASK = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _purpose_for_phase(phase: str) -> str:
    purposes = {
        "train": "online_training",
        "dev": "development_evaluation",
        "final_test": "final_evaluation",
    }
    try:
        return purposes[phase]
    except KeyError as exc:
        raise ValueError(f"Unsupported M4 phase {phase!r}") from exc


@dataclass(frozen=True)
class M4RunConfig:
    """Resolved settings shared by every method/seed in the primary matrix."""

    algorithm_id: str
    seed: int
    phase: Literal["train", "dev", "final_test"]
    base_model: str = "/data/share/model/Qwen3.5-4B"
    adapter_rank: int = 8
    max_environment_steps: int = 20
    max_model_turns: int = 20
    group_size: int = ONLINE_GROUP_SIZE
    online_passes: int = ONLINE_PASSES
    collected_action_token_cap: int = COLLECTED_ACTION_TOKEN_CAP
    eval_rollouts_per_task: int = EVAL_ROLLOUTS_PER_TASK
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    raw_sampling_logprob_tolerance: float = 0.05

    @property
    def algorithm(self) -> AlgorithmSpec:
        return get_algorithm_spec(self.algorithm_id)

    @property
    def split(self) -> str:
        return {"train": "train", "dev": "dev", "final_test": "test"}[self.phase]

    def validate(self, *, task_root: Path = DEFAULT_TASK_ROOT) -> dict[str, Any]:
        """Check fairness constants and deny any attempted final-test update."""
        if self.seed not in STUDY_SEEDS:
            raise ValueError(f"M4 seed must be one of {STUDY_SEEDS}, got {self.seed}")
        if self.adapter_rank <= 0:
            raise ValueError("adapter_rank must be positive")
        if self.max_environment_steps <= 0 or self.max_model_turns <= 0:
            raise ValueError("step limits must be positive")
        if self.collected_action_token_cap <= 0:
            raise ValueError("collected_action_token_cap must be positive")
        if self.eval_rollouts_per_task <= 0:
            raise ValueError("eval_rollouts_per_task must be positive")
        if self.algorithm.regime == "online" and self.phase == "train":
            if self.group_size != ONLINE_GROUP_SIZE:
                raise ValueError(f"online group_size is fixed at {ONLINE_GROUP_SIZE}")
            if self.online_passes != ONLINE_PASSES:
                raise ValueError(f"online_passes is fixed at {ONLINE_PASSES}")
            if (self.temperature, self.top_p, self.top_k) != (1.0, 1.0, 0):
                raise ValueError("online M4 collection requires temperature=1, top_p=1, top_k=0")
        if self.phase == "train":
            # Offline SFT/RSFT and online updates both use only the train
            # source.  Non-train phases are evaluation/model-selection only.
            purpose = "online_training"
        else:
            purpose = _purpose_for_phase(self.phase)
        task_dir = Path(task_root).expanduser().resolve() / self.split
        split_manifest = assert_m4_split_purpose(task_dir, purpose)
        expected_task_count = SPLIT_WORLD_COUNTS[self.split] * 4
        if split_manifest.get("task_count") != expected_task_count:
            raise ValueError(
                f"M4 split task count changed: expected {expected_task_count}, "
                f"got {split_manifest.get('task_count')}"
            )
        return split_manifest


def build_m4_run_manifest(
    config: M4RunConfig,
    *,
    task_root: Path = DEFAULT_TASK_ROOT,
    seed_dir: Path = DEFAULT_SEED_DIR,
) -> dict[str, Any]:
    """Produce the immutable provenance payload before a job may start."""
    task_root = Path(task_root).expanduser().resolve()
    seed_dir = Path(seed_dir).expanduser().resolve()
    split_manifest = config.validate(task_root=task_root)
    validation = validate_m4_rlvr_dataset(task_root, seed_dir=seed_dir)
    if not validation.get("valid"):
        raise ValueError(f"M4 dataset validation failed: {validation.get('errors')}")
    task_dir = task_root / config.split
    task_files = sorted(task_dir.glob("*.jsonl"))
    if len(task_files) != 2:
        raise ValueError(f"Expected one public and one oracle file in {task_dir}")
    return {
        "schema_version": PROTOCOL_VERSION,
        "study_dataset_id": DATASET_ID,
        "git_sha": _git_sha(),
        "algorithm": asdict(config.algorithm),
        "config": asdict(config),
        "resolved_split": config.split,
        "task_dir": str(task_dir),
        "seed_dir": str(seed_dir),
        "split_manifest": split_manifest,
        "hashes": {
            "dataset_manifest_sha256": _sha256(task_root / "dataset_manifest.json"),
            "split_manifest_sha256": _sha256(task_dir / "m4_split_manifest.json"),
            "seed_manifest_sha256": _sha256(seed_dir / "manifest.json"),
            **{path.name: _sha256(path) for path in task_files},
        },
    }


def write_m4_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically persist a preflight manifest; never overwrite a different one."""
    path = Path(path).expanduser().resolve()
    content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Refusing to overwrite a different M4 manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
