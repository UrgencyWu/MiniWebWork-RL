"""Machine-checkable M4 study configuration and provenance manifest."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

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
M4_MAX_ENVIRONMENT_STEPS = 20
M4_MAX_MODEL_TURNS = 20
M4_MAX_NEW_TOKENS = 128
M4_TRAIN_TASK_COUNT = SPLIT_WORLD_COUNTS["train"] * 4
ONLINE_PASS_ACTION_TOKEN_CAP = COLLECTED_ACTION_TOKEN_CAP // ONLINE_PASSES
if ONLINE_PASS_ACTION_TOKEN_CAP * ONLINE_PASSES != COLLECTED_ACTION_TOKEN_CAP:
    raise RuntimeError("M4 action-token cap must divide evenly across ordered passes")
RSFT_TRAIN_TASKS_PER_PASS = ONLINE_PASS_ACTION_TOKEN_CAP // (
    ONLINE_GROUP_SIZE * M4_MAX_MODEL_TURNS * M4_MAX_NEW_TOKENS
)
if RSFT_TRAIN_TASKS_PER_PASS <= 0:
    raise RuntimeError("M4 RSFT fixed roster must contain at least one complete K-way group")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def m4_task_source_sha256(task_dir: Path, split: str) -> str:
    """Hash the exact public/oracle pair consumed by the strict collector.

    The rollout collector deliberately binds both files: public tasks define
    the episode order while the oracle is loaded by the environment during
    verification.  Keep the filenames and ordering identical to
    ``m2_3_mini_single_probe._load_tasks`` so manifests and artifacts can be
    checked without an ambiguous, public-only proxy hash.
    """
    task_dir = Path(task_dir).expanduser().resolve()
    public_path = task_dir / f"{split}_public.jsonl"
    oracle_path = task_dir / f"{split}_oracle.jsonl"
    if not public_path.is_file() or not oracle_path.is_file():
        raise FileNotFoundError(
            f"Expected {public_path.name} and {oracle_path.name} in {task_dir}"
        )
    digest = hashlib.sha256()
    for path in (public_path, oracle_path):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def m4_task_roster_sha256(task_ids: Iterable[str]) -> str:
    """Return a canonical hash of a task-ID roster used for audit joins."""
    task_ids = sorted(task_ids)
    if len(task_ids) != len(set(task_ids)) or any(not task_id for task_id in task_ids):
        raise ValueError("M4 task roster must contain unique non-empty task IDs")
    return hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()


def m4_rsft_train_task_roster(task_root: Path, seed: int) -> tuple[str, ...]:
    """Derive the fixed, bounded RSFT train roster from the frozen public set."""
    if seed not in STUDY_SEEDS:
        raise ValueError(f"M4 seed must be one of {STUDY_SEEDS}, got {seed}")
    public_path = Path(task_root).expanduser().resolve() / "train" / "train_public.jsonl"
    if not public_path.is_file():
        raise FileNotFoundError(f"M4 RSFT train public file not found: {public_path}")
    task_ids = [
        row.get("task_id")
        for raw_line in public_path.read_text(encoding="utf-8").splitlines()
        if raw_line.strip()
        for row in [json.loads(raw_line)]
    ]
    if (
        len(task_ids) != M4_TRAIN_TASK_COUNT
        or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
        or len(task_ids) != len(set(task_ids))
    ):
        raise ValueError("frozen M4 RSFT train roster does not contain the declared 240 unique task IDs")
    random.Random(seed).shuffle(task_ids)
    return tuple(task_ids[:RSFT_TRAIN_TASKS_PER_PASS])


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
    max_environment_steps: int = M4_MAX_ENVIRONMENT_STEPS
    max_model_turns: int = M4_MAX_MODEL_TURNS
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
        if self.collected_action_token_cap != COLLECTED_ACTION_TOKEN_CAP:
            raise ValueError(f"collected_action_token_cap is fixed at {COLLECTED_ACTION_TOKEN_CAP}")
        if self.eval_rollouts_per_task != EVAL_ROLLOUTS_PER_TASK:
            raise ValueError(f"eval_rollouts_per_task is fixed at {EVAL_ROLLOUTS_PER_TASK}")
        strict_train_collection = self.phase == "train" and (
            self.algorithm.regime == "online" or self.algorithm_id == "rsft"
        )
        if strict_train_collection:
            if self.group_size != ONLINE_GROUP_SIZE:
                raise ValueError(f"strict train group_size is fixed at {ONLINE_GROUP_SIZE}")
            if self.online_passes != ONLINE_PASSES:
                raise ValueError(f"strict train online_passes is fixed at {ONLINE_PASSES}")
            if self.max_model_turns != M4_MAX_MODEL_TURNS:
                raise ValueError(f"strict train max_model_turns is fixed at {M4_MAX_MODEL_TURNS}")
            if self.max_environment_steps != M4_MAX_ENVIRONMENT_STEPS:
                raise ValueError(
                    f"strict train max_environment_steps is fixed at {M4_MAX_ENVIRONMENT_STEPS}"
                )
            if (self.temperature, self.top_p, self.top_k) != (1.0, 1.0, 0):
                raise ValueError("strict M4 collection requires temperature=1, top_p=1, top_k=0")
        if self.phase == "train":
            # Both regimes use only train, but the split manifest keeps their
            # data lineage distinct so an offline control cannot be mistaken
            # for an online policy update.
            purpose = (
                "online_training"
                if self.algorithm.regime == "online"
                else "offline_training"
            )
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
            "task_source_sha256": m4_task_source_sha256(task_dir, config.split),
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
