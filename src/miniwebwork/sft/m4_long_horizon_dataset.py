"""Build unique, verified SFT turn evidence for the focused long-horizon study."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

from ..agent_env.environment import ProcurementBrowserEnv
from ..data_generation.expert_agent import OracleExpertProcurementAgent
from ..data_generation.m4_long_horizon import (
    DATASET_ID as TASK_DATASET_ID,
    DEFAULT_OUTPUT_DIR as DEFAULT_TASK_ROOT,
    DEFAULT_SEED_DIR,
    SPLIT_MANIFEST_FILENAME,
    assert_long_horizon_split_purpose,
)
from ..model_agent import prompt_builder
from ..tasks import get_oracle, load_public_tasks

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sft" / "m4_long_horizon_verified_v1"
SFT_DATASET_ID = "m4_long_horizon_verified_sft_v1"
SFT_SCHEMA = "m4_long_horizon_verified_sft_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _chat_template_token_ids(value: Any, *, field: str) -> list[int]:
    """Normalize one tokenizer chat-template result to a single token-id list."""

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(f"{field} chat template mapping has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token_id, int) for token_id in value):
        raise ValueError(f"{field} chat template did not return one token-id list")
    return value


def completion_only_token_statistics(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
    max_length: int = 6144,
) -> dict[str, int | bool]:
    """Return the exact labels produced by prefix-masked chat-template training.

    The completion includes assistant terminators emitted by the model chat
    template.  Re-tokenizing only the raw JSON action would silently undercount
    the labels that the trainer actually optimizes.
    """

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("completion-only example must end with one assistant message")
    kwargs = dict(chat_template_kwargs or {})
    prompt_ids = _chat_template_token_ids(
        tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        ),
        field="prompt",
    )
    full_ids = _chat_template_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            **kwargs,
        ),
        field="full",
    )
    if len(full_ids) < len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant completion is not a strict continuation of the prompt template")
    untruncated_labels = len(full_ids) - len(prompt_ids)
    forward_tokens = min(len(full_ids), max_length)
    effective_labels = max(0, forward_tokens - min(len(prompt_ids), max_length))
    return {
        "prompt_tokens": len(prompt_ids),
        "untruncated_forward_tokens": len(full_ids),
        "untruncated_completion_label_tokens": untruncated_labels,
        "forward_tokens": forward_tokens,
        "effective_completion_label_tokens": effective_labels,
        "truncated": len(full_ids) > max_length,
    }


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


def normalize_reference_action(observation, action) -> dict[str, Any]:
    """Normalize a live action to the frozen reference-trace representation."""

    result: dict[str, Any] = {"action": action.action}
    if action.target:
        testids = [
            element.testid
            for element in observation.elements
            if element.element_id == action.target
        ]
        if len(testids) != 1 or not testids[0]:
            raise ValueError(f"action target does not map to one visible testid: {action.target}")
        result["target_testid"] = testids[0]
    if action.value:
        result["value"] = action.value
    if action.checked is not None:
        result["checked"] = action.checked
    return result


def build_verified_sft_split(
    output_dir: Path,
    *,
    task_dir: Path,
    seed_dir: Path = DEFAULT_SEED_DIR,
    max_steps: int = 20,
    task_limit: int | None = None,
    purpose: str,
    sample_split: str,
    output_filename: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    task_dir = Path(task_dir).expanduser().resolve()
    seed_dir = Path(seed_dir).expanduser().resolve()
    if max_steps != 20:
        raise ValueError("focused SFT reference replay is frozen at 20 max steps")
    if task_limit is not None and task_limit <= 0:
        raise ValueError("task_limit must be positive")
    if output_filename != Path(output_filename).name or not output_filename.endswith(".jsonl"):
        raise ValueError("output_filename must be a bare JSONL filename")
    split_manifest = assert_long_horizon_split_purpose(task_dir, purpose)
    tasks = sorted(load_public_tasks(task_dir), key=lambda task: task["task_id"])
    if task_limit is not None:
        tasks = tasks[:task_limit]
    if not tasks:
        raise ValueError("verified SFT split requires at least one task")

    records: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    with ProcurementBrowserEnv(
        max_steps=max_steps,
        run_id=f"m4_long_horizon_sft_{sample_split}",
        headless=True,
        keep_db=False,
        task_dir=task_dir,
        seed_dir=seed_dir,
    ) as environment:
        environment.set_agent_name("m4_long_horizon_oracle_expert")
        for task in tasks:
            task_id = task["task_id"]
            oracle = get_oracle(task_id, task_dir=task_dir)
            if oracle is None:
                raise ValueError(f"missing long-horizon oracle: {task_id}")
            observation = environment.reset(task_id)
            expert = OracleExpertProcurementAgent(oracle, max_steps=max_steps)
            history: list[dict[str, Any]] = []
            normalized_trace: list[dict[str, Any]] = []
            terminal = None
            for turn_index in range(1, max_steps + 1):
                messages = prompt_builder.build_messages(observation, history)
                action_object = expert.act(observation)
                action = action_object.to_dict()
                normalized_trace.append(
                    normalize_reference_action(observation, action_object)
                )
                target = action.get("target")
                if (
                    isinstance(target, str)
                    and target
                    and target not in prompt_builder.visible_element_ids(observation)
                ):
                    raise RuntimeError(
                        "compact prompt hid reference action target: "
                        f"task={task_id} turn={turn_index} target={target}"
                    )
                records.append(
                    {
                        "sample_id": f"{task_id}:turn:{turn_index}",
                        "dataset_id": SFT_DATASET_ID,
                        "split": sample_split,
                        "task_id": task_id,
                        "task_type": task.get("task_type", "unknown"),
                        "task_family": task.get("task_family", "unknown"),
                        "horizon_stratum": task.get("horizon_stratum", "unknown"),
                        "source": "m4_long_horizon_verified_browser_replay",
                        "turn_index": turn_index,
                        "messages": messages
                        + [
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    action,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                            }
                        ],
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
                    f"long-horizon reference replay failed: task={task_id} terminal={terminal!r}"
                )
            if normalized_trace != oracle.get("reference_trace"):
                raise RuntimeError(f"reference trace drift during SFT replay: {task_id}")
            if len(normalized_trace) != oracle.get("oracle_min_env_actions"):
                raise RuntimeError(f"reference horizon drift during SFT replay: {task_id}")
            trace_hash = hashlib.sha256(
                json.dumps(
                    normalized_trace,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if trace_hash != oracle.get("oracle_trace_sha256"):
                raise RuntimeError(f"reference trace hash drift during SFT replay: {task_id}")
            task_summaries.append(
                {
                    "task_id": task_id,
                    "task_family": task.get("task_family"),
                    "horizon_stratum": task.get("horizon_stratum"),
                    "turn_count": len(normalized_trace),
                    "oracle_trace_sha256": trace_hash,
                    "verifier_success": True,
                }
            )

    sample_ids = [record["sample_id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("verified SFT split contains duplicate sample IDs")
    records_text = _jsonl_text(records)
    manifest = {
        "schema_version": SFT_SCHEMA,
        "dataset_id": SFT_DATASET_ID,
        "builder_git_sha": _git_sha(),
        "task_source_dataset_id": TASK_DATASET_ID,
        "task_split": split_manifest["split"],
        "purpose": purpose,
        "sample_split": sample_split,
        "task_count": len(tasks),
        "sample_count": len(records),
        "unique_sample_count": len(set(sample_ids)),
        "duplicate_sample_count": 0,
        "all_reference_trajectories_verified": True,
        "max_steps": max_steps,
        "prompt_contract": prompt_builder.PROMPT_VERSION,
        "prompt_system_sha256": prompt_builder.prompt_sha256(),
        "context_contract": prompt_builder.context_contract(),
        "task_split_manifest_sha256": _sha256(task_dir / SPLIT_MANIFEST_FILENAME),
        "seed_manifest_sha256": _sha256(seed_dir / "manifest.json"),
        "output_filename": output_filename,
        "records_sha256": hashlib.sha256(records_text.encode("utf-8")).hexdigest(),
        "completion_label_contract": "one assistant action completion per unique turn; no repeated packing",
        "task_summaries": task_summaries,
    }
    _atomic_write(output_dir / output_filename, records_text)
    _atomic_write(
        output_dir / f"{sample_split}_manifest.json",
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return manifest


def build_verified_sft_corpus(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    task_root: Path = DEFAULT_TASK_ROOT,
    seed_dir: Path = DEFAULT_SEED_DIR,
    max_steps: int = 20,
    train_task_limit: int | None = None,
    dev_task_limit: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    task_root = Path(task_root).expanduser().resolve()
    train = build_verified_sft_split(
        output_dir,
        task_dir=task_root / "train",
        seed_dir=seed_dir,
        max_steps=max_steps,
        task_limit=train_task_limit,
        purpose="offline_training",
        sample_split="train",
        output_filename="train.jsonl",
    )
    dev = build_verified_sft_split(
        output_dir,
        task_dir=task_root / "dev",
        seed_dir=seed_dir,
        max_steps=max_steps,
        task_limit=dev_task_limit,
        purpose="model_selection",
        sample_split="dev",
        output_filename="valid.jsonl",
    )
    manifest = {
        "schema_version": SFT_SCHEMA,
        "dataset_id": SFT_DATASET_ID,
        "builder_git_sha": _git_sha(),
        "task_source_dataset_id": TASK_DATASET_ID,
        "task_dataset_manifest_sha256": _sha256(task_root / "dataset_manifest.json"),
        "seed_manifest_sha256": _sha256(Path(seed_dir) / "manifest.json"),
        "prompt_contract": prompt_builder.PROMPT_VERSION,
        "prompt_system_sha256": prompt_builder.prompt_sha256(),
        "context_contract": prompt_builder.context_contract(),
        "train": train,
        "dev": dev,
        "train_sha256": _sha256(output_dir / "train.jsonl"),
        "valid_sha256": _sha256(output_dir / "valid.jsonl"),
        "optimizer_boundary": "train.jsonl only; valid.jsonl never enters optimizer",
        "repetition_policy": "none",
    }
    _atomic_write(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return manifest


def validate_verified_sft_corpus(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    task_root: Path = DEFAULT_TASK_ROOT,
    seed_dir: Path = DEFAULT_SEED_DIR,
    require_full_roster: bool = True,
) -> dict[str, Any]:
    """Validate unique split evidence and its task/reference lineage."""

    output_dir = Path(output_dir).expanduser().resolve()
    task_root = Path(task_root).expanduser().resolve()
    seed_dir = Path(seed_dir).expanduser().resolve()
    errors: list[str] = []
    try:
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [str(exc)]}
    if manifest.get("schema_version") != SFT_SCHEMA or manifest.get("dataset_id") != SFT_DATASET_ID:
        errors.append("SFT corpus schema or dataset id mismatch")
    if manifest.get("task_source_dataset_id") != TASK_DATASET_ID:
        errors.append("SFT task source dataset id mismatch")
    if manifest.get("prompt_contract") != prompt_builder.PROMPT_VERSION:
        errors.append("SFT prompt contract mismatch")
    if manifest.get("repetition_policy") != "none":
        errors.append("SFT corpus repetition policy is not none")
    expected_task_manifest_hash = _sha256(task_root / "dataset_manifest.json")
    expected_seed_manifest_hash = _sha256(seed_dir / "manifest.json")
    if manifest.get("task_dataset_manifest_sha256") != expected_task_manifest_hash:
        errors.append("SFT task dataset manifest hash mismatch")
    if manifest.get("seed_manifest_sha256") != expected_seed_manifest_hash:
        errors.append("SFT seed manifest hash mismatch")

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_task_ids: dict[str, set[str]] = {}
    for split, filename, expected_tasks in (
        ("train", "train.jsonl", 240),
        ("dev", "valid.jsonl", 72),
    ):
        path = output_dir / filename
        expected_hash = manifest.get(f"{'train' if split == 'train' else 'valid'}_sha256")
        if not path.is_file() or expected_hash != _sha256(path):
            errors.append(f"{split} records hash mismatch")
            continue
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except json.JSONDecodeError as exc:
            errors.append(f"{split} JSONL is invalid: {exc}")
            continue
        split_rows[split] = rows
        sample_ids = [row.get("sample_id") for row in rows]
        if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
            errors.append(f"{split} contains invalid sample ids")
        if len(sample_ids) != len(set(sample_ids)):
            errors.append(f"{split} contains duplicate samples")
        if any(row.get("split") != split for row in rows):
            errors.append(f"{split} JSONL contains wrong split rows")
        if any(
            not isinstance(row.get("messages"), list)
            or not row["messages"]
            or row["messages"][-1].get("role") != "assistant"
            for row in rows
        ):
            errors.append(f"{split} contains rows without one assistant completion")
        task_ids = {row.get("task_id") for row in rows}
        split_task_ids[split] = task_ids
        split_manifest = manifest.get(split, {})
        if split_manifest.get("sample_count") != len(rows):
            errors.append(f"{split} sample count mismatch")
        if split_manifest.get("unique_sample_count") != len(set(sample_ids)):
            errors.append(f"{split} unique sample count mismatch")
        if split_manifest.get("duplicate_sample_count") != 0:
            errors.append(f"{split} duplicate count is nonzero")
        if split_manifest.get("all_reference_trajectories_verified") is not True:
            errors.append(f"{split} reference verification is not complete")
        if split_manifest.get("task_count") != len(task_ids):
            errors.append(f"{split} task coverage disagrees with rows")
        if require_full_roster and len(task_ids) != expected_tasks:
            errors.append(f"{split} does not cover the full {expected_tasks}-task roster")
        oracle_rows = {
            row["task_id"]: row
            for row in [
                json.loads(line)
                for line in (task_root / split / f"{split}_oracle.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
        }
        summaries = split_manifest.get("task_summaries", [])
        if len(summaries) != len(task_ids):
            errors.append(f"{split} task summaries do not cover the roster")
        for summary in summaries:
            task_id = summary.get("task_id")
            oracle = oracle_rows.get(task_id)
            if oracle is None:
                errors.append(f"{split} summary references unknown task: {task_id}")
                continue
            if summary.get("verifier_success") is not True:
                errors.append(f"{task_id} reference verifier did not pass")
            if summary.get("turn_count") != oracle.get("oracle_min_env_actions"):
                errors.append(f"{task_id} turn count does not match oracle horizon")
            if summary.get("oracle_trace_sha256") != oracle.get("oracle_trace_sha256"):
                errors.append(f"{task_id} trace hash does not match oracle")
    if split_task_ids.get("train", set()) & split_task_ids.get("dev", set()):
        errors.append("train/dev task overlap in SFT corpus")
    return {
        "valid": not errors,
        "errors": errors,
        "dataset_id": manifest.get("dataset_id"),
        "builder_git_sha": manifest.get("builder_git_sha"),
        "train_sample_count": len(split_rows.get("train", [])),
        "dev_sample_count": len(split_rows.get("dev", [])),
        "train_task_count": len(split_task_ids.get("train", set())),
        "dev_task_count": len(split_task_ids.get("dev", set())),
        "train_sha256": manifest.get("train_sha256"),
        "valid_sha256": manifest.get("valid_sha256"),
    }
