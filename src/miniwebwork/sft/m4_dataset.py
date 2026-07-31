"""Build canonical M4 oracle-SFT action examples through the real browser UI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ..agent_env.environment import ProcurementBrowserEnv
from ..data_generation.expert_agent import OracleExpertProcurementAgent
from ..data_generation.m4_rlvr import assert_m4_split_purpose
from ..model_agent import prompt_builder
from ..tasks import get_oracle, load_public_tasks

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASK_DIR = PROJECT_ROOT / "data" / "tasks" / "m4_rlvr_v1" / "train"
DEFAULT_SEED_DIR = PROJECT_ROOT / "data" / "seed_m4_rlvr_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sft" / "m4_oracle_v1"
DATASET_ID = "m4_oracle_sft_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_text(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _feedback(action: dict[str, Any], step_result, model_turn_index: int) -> dict[str, Any]:
    raw = step_result.info.get("action_result", {}) if step_result is not None else {}
    return {
        "model_turn_index": model_turn_index,
        "action": action,
        "parse_ok": True,
        "result": {
            "success": bool(raw.get("success", False)),
            "error_code": str(raw.get("error_code", ""))[:100],
            "message": str(raw.get("message", ""))[:200],
            "page_changed": bool(raw.get("page_changed", False)),
        },
        "page_type": (
            step_result.observation.page_type
            if step_result is not None and step_result.observation is not None
            else "unknown"
        ),
    }


def build_m4_oracle_sft_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    task_dir: Path = DEFAULT_TASK_DIR,
    seed_dir: Path = DEFAULT_SEED_DIR,
    max_steps: int = 20,
    task_limit: int | None = None,
    purpose: str = "offline_training",
    sample_split: str = "train",
    output_filename: str = "train.jsonl",
) -> dict[str, Any]:
    """Replay all requested train oracle trajectories and emit SFT JSONL.

    The browser state, prompt builder and feedback history are the same as
    model inference.  This prevents a compacted, teacher-only prompt dialect
    from becoming an accidental advantage for the SFT control baseline.
    """
    output_dir = output_dir.expanduser().resolve()
    task_dir = task_dir.expanduser().resolve()
    seed_dir = seed_dir.expanduser().resolve()
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if task_limit is not None and task_limit <= 0:
        raise ValueError("task_limit must be positive")
    if output_filename != Path(output_filename).name or not output_filename.endswith(".jsonl"):
        raise ValueError("output_filename must be a bare .jsonl filename")
    split_manifest = assert_m4_split_purpose(task_dir, purpose)
    tasks = sorted(load_public_tasks(task_dir), key=lambda task: task["task_id"])
    if task_limit is not None:
        tasks = tasks[:task_limit]
    if not tasks:
        raise ValueError("M4 SFT dataset requires at least one task")

    records: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    run_id = "m4_oracle_sft_builder"
    with ProcurementBrowserEnv(
        max_steps=max_steps,
        run_id=run_id,
        headless=True,
        keep_db=False,
        task_dir=task_dir,
        seed_dir=seed_dir,
    ) as environment:
        environment.set_agent_name("m4_oracle_sft_builder")
        for task in tasks:
            task_id = task["task_id"]
            oracle = get_oracle(task_id, task_dir=task_dir)
            if oracle is None:
                raise ValueError(f"Missing M4 oracle for {task_id}")
            observation = environment.reset(task_id)
            expert = OracleExpertProcurementAgent(oracle, max_steps=max_steps)
            history: list[dict[str, Any]] = []
            terminal = None
            for turn_index in range(1, max_steps + 1):
                messages = prompt_builder.build_messages(observation, history)
                action_object = expert.act(observation)
                action = action_object.to_dict()
                records.append(
                    {
                        "sample_id": f"{task_id}:turn:{turn_index}",
                        "dataset_id": DATASET_ID,
                        "split": sample_split,
                        "task_id": task_id,
                        "task_type": task.get("task_type", "unknown"),
                        "source": "m4_oracle_browser_replay",
                        "turn_index": turn_index,
                        "messages": messages + [{"role": "assistant", "content": json.dumps(action, ensure_ascii=False, sort_keys=True)}],
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                )
                terminal = environment.step(action_object)
                history.append(_feedback(action, terminal, turn_index))
                if terminal.observation is not None:
                    observation = terminal.observation
                if terminal.terminated or terminal.truncated:
                    break
            if terminal is None or terminal.truncated or terminal.reward != 1.0:
                raise RuntimeError(
                    f"M4 oracle browser replay failed for {task_id}: "
                    f"terminal={terminal!r}"
                )
            task_summaries.append(
                {"task_id": task_id, "turn_count": len(history), "verifier_success": True}
            )

    train_text = _jsonl_text(records)
    manifest = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "task_source_dataset_id": split_manifest["dataset_id"],
        "task_split": split_manifest["split"],
        "purpose": purpose,
        "sample_split": sample_split,
        "task_count": len(tasks),
        "sample_count": len(records),
        "max_steps": max_steps,
        "prompt_contract": getattr(prompt_builder, "PROMPT_VERSION", "unknown"),
        "task_split_manifest_sha256": _sha256(task_dir / "m4_split_manifest.json"),
        "seed_manifest_sha256": _sha256(seed_dir / "manifest.json"),
        "output_filename": output_filename,
        "records_sha256": hashlib.sha256(train_text.encode("utf-8")).hexdigest(),
        "task_summaries": task_summaries,
    }
    _atomic_write(output_dir / output_filename, train_text)
    _atomic_write(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return manifest


def build_m4_oracle_sft_corpus(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    train_task_dir: Path = DEFAULT_TASK_DIR,
    dev_task_dir: Path = PROJECT_ROOT / "data" / "tasks" / "m4_rlvr_v1" / "dev",
    seed_dir: Path = DEFAULT_SEED_DIR,
    max_steps: int = 20,
    train_task_limit: int | None = None,
    dev_task_limit: int | None = None,
) -> dict[str, Any]:
    """Build train-only supervision plus a dev-only model-selection corpus."""
    output_dir = output_dir.expanduser().resolve()
    if train_task_limit is not None and train_task_limit <= 0:
        raise ValueError("train_task_limit must be positive")
    if dev_task_limit is not None and dev_task_limit <= 0:
        raise ValueError("dev_task_limit must be positive")
    train = build_m4_oracle_sft_dataset(
        output_dir,
        task_dir=train_task_dir,
        seed_dir=seed_dir,
        max_steps=max_steps,
        task_limit=train_task_limit,
        purpose="offline_training",
        sample_split="train",
        output_filename="train.jsonl",
    )
    dev = build_m4_oracle_sft_dataset(
        output_dir,
        task_dir=dev_task_dir,
        seed_dir=seed_dir,
        max_steps=max_steps,
        task_limit=dev_task_limit,
        purpose="model_selection",
        sample_split="dev",
        output_filename="valid.jsonl",
    )
    manifest = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "train": train,
        "dev": dev,
        "train_sha256": _sha256(output_dir / "train.jsonl"),
        "valid_sha256": _sha256(output_dir / "valid.jsonl"),
        "selection_boundary": "dev examples are evaluation/model-selection only, never optimizer samples",
    }
    _atomic_write(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return manifest
