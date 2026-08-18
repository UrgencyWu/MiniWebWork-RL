#!/usr/bin/env python3
"""Collect or resume M6 Raw success data and shared K4/K8 rollouts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.adapter_view import build_vllm_adapter_view  # noqa: E402
from miniwebwork.long_horizon_rl.contracts import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    directory_sha256,
    sha256_json,
)
from miniwebwork.long_horizon_rl.model_manifest import (  # noqa: E402
    BASE_MODEL_MANIFEST_PATH,
    validate_base_model_manifest,
)
from miniwebwork.long_horizon_rl.vllm_backend import (  # noqa: E402
    AsyncVLLMGenerationEngine,
    RawAsyncVLLMGenerationEngine,
    RawVLLMBackendConfig,
    RolloutRequestContext,
    ThreadsafeVLLMBackend,
)
from miniwebwork.m6_posttraining_protocol import (  # noqa: E402
    load_protocol,
    validate_split_lock,
)
from miniwebwork.m6_pilot import validate_artifact_git_compatibility  # noqa: E402
from miniwebwork.model_agent.agent_loop import run_model_episode  # noqa: E402
from miniwebwork.model_agent.model_backend import GenerationResult  # noqa: E402
from miniwebwork.webshop_rl.agent import QwenWebShopAgent  # noqa: E402
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import (  # noqa: E402
    build_committed_group,
    trajectory_from_episode,
    validate_committed_group,
)
from miniwebwork.webshop_rl.online_training import MAX_NEW_TOKENS, M5VLLMBackendConfig  # noqa: E402
from miniwebwork.webshop_rl.verifier_td import (  # noqa: E402
    annotate_episode_with_verifier,
    public_state_anchor_signature,
)

ATTEMPT_SCHEMA = "m6_rollout_attempt_v1"
ATTEMPT_START_SCHEMA = "m6_rollout_attempt_start_v1"
REPORT_SCHEMA = "m6_rollout_collection_report_v1"
MAX_INFRASTRUCTURE_ATTEMPTS = 4
LINEAGE_FIELDS = (
    "adapter_sha256",
    "rollout_adapter_sha256",
    "adapter_semantic_sha256",
)
PHASE4_TEACHER_MODELS = {
    Path("/data/share/model/Qwen3.5-9B").resolve(),
    Path("/data/share/model/Qwen3.6-35B-A3B-FP8").resolve(),
}
PHASE10_STUDENT_MODEL = Path("/data/share/model/Qwen3.5-4B").resolve()
PHASE10_TEACHER_MODEL = Path("/data/share/model/Qwen3.6-35B-A3B-FP8").resolve()
PHASE10B_STUDENT_ADAPTER = Path(
    "/home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/mini/pilot_sft/final_adapter"
).resolve()
PHASE10B_QUALIFICATION_MODELS = {
    "student": PHASE10_STUDENT_MODEL,
    "S_nav": Path("/data/share/model/Qwen3.5-9B").resolve(),
    "S_match": Path("/data/share/model/Qwen3.5-35B-A3B").resolve(),
    "S_finish": Path("/data/share/model/Qwen3.6-35B-A3B-FP8").resolve(),
}
PHASE10B_QUALIFICATION_TASK_COUNTS = {"student": 24, "S_nav": 24, "S_match": 24, "S_finish": 24}
PHASE10B_OPD_SMOKE_TASK_COUNT = 8
PHASE10C_TEACHER_MODEL = Path("/data/share/model/Qwen3.5-35B-A3B").resolve()
PHASE10C_SFT35_ADAPTER = Path(
    "/home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/"
    "phase10c_qwen35_sft_specialist_opd_v1/teacher_self_sft_v1/weighted_lora_v1/final_adapter"
).resolve()
PHASE10C_SFT35_MERGED_MODEL = Path(
    "/home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/"
    "phase10c_qwen35_sft_specialist_opd_v1/teacher_self_sft_v2_4b_paradigm/merged_model_v1/model"
).resolve()
PHASE10D_SFT35_D4_MERGED_MODEL = Path(
    "/home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/"
    "phase10d_same_corpus_scale_v1/s35_d4/formal_v1/merged_model_v1/model"
).resolve()
PHASE10D_S4_D35_ADAPTER = Path(
    "/home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/"
    "phase10d_same_corpus_scale_v1/s4_d35/formal_v1/final_adapter"
).resolve()
PHASE10D_QWEN38_27B_MODEL = Path("/data/share/model/Qwen3.8-27B").resolve()
PHASE10C_EVALUATION_MODELS = {
    "raw35": PHASE10C_TEACHER_MODEL,
    "sft35": PHASE10C_SFT35_MERGED_MODEL,
    "sft35_d4": PHASE10D_SFT35_D4_MERGED_MODEL,
    "sft35_d35": PHASE10C_SFT35_MERGED_MODEL,
    "sft4": PHASE10_STUDENT_MODEL,
    "sft4_d35": PHASE10_STUDENT_MODEL,
    "raw27": PHASE10D_QWEN38_27B_MODEL,
}
PHASE10C_EVALUATION_TASK_COUNT = 96
PHASE10C_TEACHER_EXPLORATION_TASK_COUNTS = {
    "phase10c_teacher_exploration": 32,
    "phase10c_teacher_exploration_scale": 128,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalized_command(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _atomic_episode(path: Path, episode: Mapping[str, Any]) -> None:
    value = dict(episode)
    value["content_sha256"] = _self_hash(value)
    atomic_write_json(path, value)


def _attempt_indices(attempts_root: Path, group_id: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(group_id)}\.a(\d+)(?:\.started)?\.json$")
    indices: set[int] = set()
    for path in attempts_root.glob(f"{group_id}.a*.json"):
        match = pattern.fullmatch(path.name)
        _require(match is not None, f"M6 attempt filename drift: {path.name}")
        value = _load_json(path)
        expected = dict(value)
        observed = expected.pop("content_sha256", None)
        _require(observed == sha256_json(expected), f"M6 attempt artifact self-hash drift: {path.name}")
        _require(value.get("group_id") == group_id, f"M6 attempt artifact group drift: {path.name}")
        _require(int(value.get("attempt_index", -1)) == int(match.group(1)), f"M6 attempt index drift: {path.name}")
        indices.add(int(match.group(1)))
    return sorted(indices)


class DurableTokenLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, payload: Mapping[str, Any]) -> None:
        line = canonical_json_bytes(dict(payload)) + b"\n"
        with self._lock, self.path.open("ab", buffering=0) as handle:
            handle.write(line)
            os.fsync(handle.fileno())

    def audit(self) -> dict[str, Any]:
        rows = []
        if self.path.is_file():
            for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise ValueError(f"M6 token ledger line {line_number} is corrupt") from exc
                _require(isinstance(row, Mapping), "M6 token ledger row is malformed")
                rows.append(row)
        return {
            "record_count": len(rows),
            "generated_action_tokens": sum(int(row["generated_action_tokens"]) for row in rows),
            "sha256": sha256_json(rows) if rows else hashlib.sha256(b"").hexdigest(),
        }


class Phase2PrefixReplayBackend:
    """Replay an audited short-horizon prefix before live continuation."""

    def __init__(
        self,
        *,
        live_backend: ThreadsafeVLLMBackend,
        source_turns: list[Mapping[str, Any]],
        lineage: Mapping[str, str],
    ):
        self._live_backend = live_backend
        self._source_turns = [dict(turn) for turn in source_turns]
        self._lineage = dict(lineage)
        self._position = 0

    @property
    def consumed_prefix_turns(self) -> int:
        return self._position

    def generate(self, messages: list[dict[str, Any]]) -> GenerationResult:
        if self._position >= len(self._source_turns):
            return self._live_backend.generate(messages)
        source = self._source_turns[self._position]
        self._position += 1
        prompt_ids = [int(value) for value in source["prompt_token_ids"]]
        generated_ids = [int(value) for value in source["generated_token_ids"]]
        behavior = [float(value) for value in source["behavior_logprobs"]]
        sampling = [float(value) for value in source["sampling_logprobs"]]
        _require(prompt_ids and generated_ids, "M6 Phase2 replay turn lacks token evidence")
        _require(
            len(generated_ids) == len(behavior) == len(sampling),
            "M6 Phase2 replay turn logprob/token drift",
        )
        return GenerationResult(
            raw_text=str(source["raw_output"]),
            new_tokens=len(generated_ids),
            input_tokens=len(prompt_ids),
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated_ids,
            logprobs=behavior,
            sampling_logprobs=sampling,
            request_id=str(source["request_id"]),
            sampling_seed=int(source["sampling_seed"]),
            generation_backend="phase2_prefix_replay_v1",
            adapter_sha256=self._lineage["adapter_sha256"],
            rollout_adapter_sha256=self._lineage["rollout_adapter_sha256"],
            adapter_semantic_sha256=self._lineage["adapter_semantic_sha256"],
        )


async def _create_identity(
    *,
    base_model: Path,
    base_model_manifest: Path,
    adapter: Path | None,
    output: Path,
    seed: int,
    tensor_parallel_size: int,
) -> tuple[RawAsyncVLLMGenerationEngine | AsyncVLLMGenerationEngine, dict[str, str]]:
    manifest = validate_base_model_manifest(
        path=base_model_manifest,
        expected_base_model=base_model,
        verify_files=True,
    )
    if adapter is None:
        engine = await RawAsyncVLLMGenerationEngine.create(
            RawVLLMBackendConfig(
                base_model=str(base_model),
                base_model_manifest_sha256=manifest["sha256"],
                base_model_functional_sha256=manifest["payload"]["functional_file_set_sha256"],
                seed=seed,
                tensor_parallel_size=tensor_parallel_size,
            )
        )
        lineage = {
            "adapter_sha256": manifest["sha256"],
            "rollout_adapter_sha256": manifest["payload"]["functional_file_set_sha256"],
            "adapter_semantic_sha256": manifest["payload"]["functional_file_set_sha256"],
        }
    else:
        canonical = adapter.expanduser().resolve()
        adapter_config = _load_json(canonical / "adapter_config.json")
        lora_rank = int(adapter_config.get("r", 0))
        _require(lora_rank in {8, 16}, "M6 LoRA rank is unsupported")
        view_audit = build_vllm_adapter_view(
            source_adapter=canonical,
            destination=output / "rollout_adapter",
            base_model=base_model,
        )
        lineage = {
            "adapter_sha256": directory_sha256(canonical),
            "rollout_adapter_sha256": view_audit["view_directory_sha256"],
            "adapter_semantic_sha256": view_audit["semantic_tensor_sha256"],
        }
        engine = await AsyncVLLMGenerationEngine.create(
            M5VLLMBackendConfig(
                base_model=str(base_model),
                adapter_path=str(canonical),
                rollout_adapter_path=str(output / "rollout_adapter"),
                seed=seed,
                tensor_parallel_size=tensor_parallel_size,
                max_lora_rank=lora_rank,
                **lineage,
            )
        )
    return engine, lineage


def _task_roster(path: Path) -> tuple[list[str], str, str, str, str]:
    value = _load_json(path)
    _require(isinstance(value, Mapping), "M6 task roster root is malformed")
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), "M6 task roster self-hash drift")
    _require(value.get("schema_version") == "m6_rl_curriculum_v1", "M6 task roster schema drift")
    task_ids = value.get("task_ids")
    _require(
        isinstance(task_ids, list)
        and task_ids
        and len(task_ids) == len(set(task_ids))
        and all(isinstance(item, str) and item for item in task_ids),
        "M6 task roster is empty or duplicated",
    )
    split_hash = value.get("source_split_lock_content_sha256")
    _require(isinstance(split_hash, str) and split_hash, "M6 task roster split binding is missing")
    protocol_hash = value.get("protocol_sha256")
    git_sha = value.get("git_sha")
    _require(isinstance(protocol_hash, str) and protocol_hash, "M6 task roster protocol binding is missing")
    _require(isinstance(git_sha, str) and git_sha, "M6 task roster Git binding is missing")
    return list(task_ids), str(observed), split_hash, protocol_hash, git_sha


def validate_diagnostic_evaluation_contract(args: argparse.Namespace) -> None:
    """Keep phase-one seen-task evaluation outside every training path."""

    _require(args.role == "mini_train" and args.task_roster is not None, "M6 diagnostic evaluation roster drift")
    _require(args.adapter is not None, "M6 diagnostic evaluation requires an SFT or RL adapter")
    _require(args.max_model_turns == 6 and args.max_environment_steps == 6, "M6 seen-task diagnostic horizon drift")
    _require(args.maximum_action_tokens is None, "M6 diagnostic evaluation cannot claim a training token budget")


def validate_phase2_horizon_contract(args: argparse.Namespace) -> None:
    """Freeze a train-role, SFT-only, paired horizon diagnostic outside training."""

    _require(args.role == "train" and args.task_roster is not None, "M6 Phase2 horizon roster drift")
    _require(args.adapter is not None, "M6 Phase2 horizon diagnostic requires the SFT adapter")
    _require(
        (args.max_model_turns, args.max_environment_steps) in {(6, 6), (18, 15)},
        "M6 Phase2 horizon arm drift",
    )
    _require(args.maximum_tasks is None and args.task_offset == 0, "M6 Phase2 horizon task slicing is forbidden")
    _require(args.maximum_action_tokens is None, "M6 Phase2 horizon diagnostic cannot claim a training token budget")
    replay_root = getattr(args, "replay_prefix_root", None)
    if replay_root is not None:
        _require(
            (args.max_model_turns, args.max_environment_steps) == (18, 15),
            "M6 Phase2 prefix replay is only valid for the full-horizon arm",
        )


def validate_phase4_data_synthesis_contract(args: argparse.Namespace) -> None:
    """Freeze full-horizon, SFT-disjoint rollout data without enabling updates."""

    _require(args.role == "train" and args.task_roster is not None, "M6 Phase4 synthesis roster drift")
    _require(args.adapter is not None, "M6 Phase4 synthesis requires the frozen SFT adapter")
    _require(
        (args.max_model_turns, args.max_environment_steps) == (18, 15),
        "M6 Phase4 synthesis must match the full evaluation horizon",
    )
    _require(args.maximum_tasks is None and args.task_offset == 0, "M6 Phase4 synthesis task slicing is forbidden")
    _require(args.maximum_action_tokens is None, "M6 Phase4 synthesis cannot claim a training token budget")
    _require(args.replay_prefix_root is None, "M6 Phase4 synthesis cannot reuse a diagnostic prefix")


def validate_phase4_online_rl_contract(args: argparse.Namespace) -> None:
    """Allow one small full-horizon on-policy K4x4 training batch."""

    _require(args.role == "train" and args.task_roster is not None, "M6 Phase4 online roster drift")
    _require(args.adapter is not None, "M6 Phase4 online collection requires the current adapter")
    _require(args.k == 4, "M6 Phase4 online collection must use K4")
    _require(
        (args.max_model_turns, args.max_environment_steps) == (18, 15),
        "M6 Phase4 online collection must use the full horizon",
    )
    _require(args.maximum_tasks == 4, "M6 Phase4 online collection must attempt four task groups")
    _require(args.task_offset >= 0 and args.task_offset % 4 == 0, "M6 Phase4 online task offset drift")
    _require(
        args.maximum_action_tokens is not None and 0 < args.maximum_action_tokens <= 75000,
        "M6 Phase4 online action-token cap drift",
    )
    _require(args.replay_prefix_root is None, "M6 Phase4 online collection cannot replay a diagnostic prefix")


def validate_phase4_tuning_evaluation_contract(args: argparse.Namespace) -> None:
    """Freeze a fresh train-role hold-aside for comparable Raw/SFT/RL evaluation."""

    _require(args.role == "train" and args.task_roster is not None, "M6 Phase4 tuning roster drift")
    _require(args.k == 4, "M6 Phase4 tuning evaluation must use K4")
    _require(
        (args.max_model_turns, args.max_environment_steps) == (18, 15),
        "M6 Phase4 tuning evaluation must use the full horizon",
    )
    _require(args.maximum_tasks is None and args.task_offset == 0, "M6 Phase4 tuning task slicing is forbidden")
    _require(args.maximum_action_tokens is None, "M6 Phase4 tuning evaluation cannot claim a training token budget")
    _require(args.replay_prefix_root is None, "M6 Phase4 tuning evaluation cannot replay a diagnostic prefix")


def validate_phase4_teacher_probe_contract(args: argparse.Namespace) -> None:
    """Freeze a larger, public-observation teacher probe outside every RL path."""

    _require(args.role == "train" and args.task_roster is not None, "M6 Phase4 teacher roster drift")
    _require(args.adapter is None, "M6 Phase4 teacher probe must not reuse the student adapter")
    _require(
        args.base_model.expanduser().resolve() in PHASE4_TEACHER_MODELS,
        "M6 Phase4 teacher model drift",
    )
    _require(
        (args.max_model_turns, args.max_environment_steps) == (18, 15),
        "M6 Phase4 teacher probe must match the full evaluation horizon",
    )
    _require(args.maximum_tasks is None and args.task_offset == 0, "M6 Phase4 teacher task slicing is forbidden")
    _require(args.maximum_action_tokens is None, "M6 Phase4 teacher probe cannot claim a training token budget")
    _require(args.replay_prefix_root is None, "M6 Phase4 teacher probe cannot reuse a diagnostic prefix")


def validate_phase9_shared_prefix_smoke_contract(args: argparse.Namespace) -> None:
    """Freeze one eight-task SFT inference smoke with a shared public prefix."""

    _require(args.role == "train" and args.task_roster is not None, "M6 Phase9 roster drift")
    _require(args.adapter is not None, "M6 Phase9 requires the frozen SFT adapter")
    _require(args.k == 4, "M6 Phase9 must use K4")
    _require(
        (args.max_model_turns, args.max_environment_steps) == (18, 15),
        "M6 Phase9 must use the full horizon",
    )
    _require(args.maximum_tasks is None and args.task_offset == 0, "M6 Phase9 task slicing is forbidden")
    _require(args.maximum_action_tokens is None, "M6 Phase9 cannot claim a training token budget")
    _require(args.replay_prefix_root is None, "M6 Phase9 cannot use the Phase2 replay source")
    _require(args.shared_prefix_manifest is not None, "M6 Phase9 shared-prefix manifest is required")


def validate_phase10_state_suffix_smoke_contract(args: argparse.Namespace) -> None:
    """Freeze the matched student/teacher K2 historical-state engineering smoke."""

    _require(args.role == "train" and args.task_roster is not None, "M6 Phase10 smoke roster drift")
    _require(args.k == 2, "M6 Phase10 smoke must use K2")
    _require(
        (args.max_model_turns, args.max_environment_steps) == (18, 15),
        "M6 Phase10 smoke must preserve the original full-episode budget",
    )
    _require(args.maximum_tasks is None and args.task_offset == 0, "M6 Phase10 smoke task slicing is forbidden")
    _require(args.maximum_action_tokens is None, "M6 Phase10 smoke cannot claim a training token budget")
    _require(args.replay_prefix_root is None, "M6 Phase10 smoke cannot use the Phase2 replay source")
    _require(args.shared_prefix_manifest is None, "M6 Phase10 smoke cannot use the Phase9 manifest")
    _require(args.state_correction_manifest is not None, "M6 Phase10 state-correction manifest is required")
    _require(args.phase10_smoke_identity in {"student", "teacher"}, "M6 Phase10 smoke identity drift")
    model = args.base_model.expanduser().resolve()
    if args.phase10_smoke_identity == "student":
        _require(args.adapter is not None, "M6 Phase10 student smoke requires pi_0")
        _require(model == PHASE10_STUDENT_MODEL, "M6 Phase10 student model drift")
    else:
        _require(args.adapter is None, "M6 Phase10 teacher smoke cannot use a student adapter")
        _require(model == PHASE10_TEACHER_MODEL, "M6 Phase10 teacher model drift")


def validate_phase10b_specialist_qualification_contract(args: argparse.Namespace) -> None:
    """Freeze paired full-horizon qualification for the three OPD candidates."""

    _require(args.role == "train" and args.task_roster is not None, "M6 Phase10-B qualification roster drift")
    _require(args.k == 4, "M6 Phase10-B qualification must use K4")
    _require(
        (args.max_model_turns, args.max_environment_steps) == (18, 15),
        "M6 Phase10-B qualification must use the full horizon",
    )
    _require(args.maximum_tasks is None and args.task_offset == 0,
             "M6 Phase10-B qualification task slicing is forbidden")
    _require(args.maximum_action_tokens is None, "M6 Phase10-B qualification cannot claim a training token budget")
    _require(args.replay_prefix_root is None, "M6 Phase10-B qualification cannot replay a diagnostic prefix")
    _require(args.shared_prefix_manifest is None and args.state_correction_manifest is None,
             "M6 Phase10-B qualification cannot use historical state manifests")
    identity = args.phase10b_qualification_identity
    _require(identity in PHASE10B_QUALIFICATION_MODELS, "M6 Phase10-B qualification identity drift")
    _require(args.base_model.expanduser().resolve() == PHASE10B_QUALIFICATION_MODELS[identity],
             "M6 Phase10-B qualification model drift")
    if identity == "student":
        _require(
            args.adapter is not None and args.adapter.expanduser().resolve() == PHASE10B_STUDENT_ADAPTER,
            "M6 Phase10-B student qualification requires frozen pi_0",
        )
        _require(args.tensor_parallel_size == 1, "M6 Phase10-B student tensor parallelism drift")
    else:
        _require(args.adapter is None, "M6 Phase10-B Specialist qualification cannot use student adapter")
        expected_tp = 2 if identity == "S_match" else 1
        _require(args.tensor_parallel_size == expected_tp, "M6 Phase10-B Specialist tensor parallelism drift")


def validate_phase10b_opd_smoke_behavior_contract(args: argparse.Namespace) -> None:
    """Freeze an eight-task pi_0 behavior batch before any Specialist query."""

    _require(args.role == "train" and args.task_roster is not None, "M6 Phase10-B OPD smoke roster drift")
    _require(args.k == 4, "M6 Phase10-B OPD smoke must use K4")
    _require(
        (args.max_model_turns, args.max_environment_steps) == (18, 15),
        "M6 Phase10-B OPD smoke must use the full horizon",
    )
    _require(args.maximum_tasks is None and args.task_offset == 0,
             "M6 Phase10-B OPD smoke task slicing is forbidden")
    _require(args.maximum_action_tokens is None, "M6 Phase10-B OPD smoke cannot claim a training token budget")
    _require(args.replay_prefix_root is None, "M6 Phase10-B OPD smoke cannot replay a diagnostic prefix")
    _require(args.shared_prefix_manifest is None and args.state_correction_manifest is None,
             "M6 Phase10-B OPD smoke cannot use historical state manifests")
    _require(args.base_model.expanduser().resolve() == PHASE10_STUDENT_MODEL,
             "M6 Phase10-B OPD smoke Student model drift")
    _require(
        args.adapter is not None and args.adapter.expanduser().resolve() == PHASE10B_STUDENT_ADAPTER,
        "M6 Phase10-B OPD smoke requires frozen pi_0",
    )
    _require(args.tensor_parallel_size == 1, "M6 Phase10-B OPD smoke tensor parallelism drift")


def validate_phase10c_teacher_exploration_contract(args: argparse.Namespace) -> None:
    """Freeze inference-only full-environment exploration by Raw Qwen3.5-35B."""

    _require(args.role == "train" and args.task_roster is not None,
             "M6 Phase10-C teacher exploration roster drift")
    expected_k = 8 if args.mode == "phase10c_teacher_exploration_scale" else 4
    _require(args.k == expected_k, f"M6 Phase10-C teacher exploration must use K{expected_k}")
    _require(
        (args.max_model_turns, args.max_environment_steps) == (18, 15),
        "M6 Phase10-C teacher exploration must use the full horizon",
    )
    _require(args.maximum_tasks is None and args.task_offset == 0,
             "M6 Phase10-C teacher exploration task slicing is forbidden")
    _require(args.maximum_action_tokens is None,
             "M6 Phase10-C teacher exploration cannot claim a training token budget")
    _require(args.replay_prefix_root is None,
             "M6 Phase10-C teacher exploration cannot replay a diagnostic prefix")
    _require(args.shared_prefix_manifest is None and args.state_correction_manifest is None,
             "M6 Phase10-C teacher exploration cannot use historical state manifests")
    _require(args.base_model.expanduser().resolve() == PHASE10C_TEACHER_MODEL,
             "M6 Phase10-C teacher exploration model drift")
    _require(args.adapter is None, "M6 Phase10-C teacher exploration must start from Raw 35B")
    _require(args.tensor_parallel_size == 2,
             "M6 Phase10-C teacher exploration tensor parallelism drift")


def validate_phase10c_teacher_stage_evaluation_contract(args: argparse.Namespace) -> None:
    """Freeze paired Raw35/SFT35/SFT4 full-environment evaluation."""

    _require(args.role == "train" and args.task_roster is not None,
             "M6 Phase10-C evaluation roster drift")
    _require(args.k == 4, "M6 Phase10-C evaluation must use K4")
    _require(
        (args.max_model_turns, args.max_environment_steps) == (18, 15),
        "M6 Phase10-C evaluation must use the full horizon",
    )
    _require(args.maximum_tasks is None and args.task_offset == 0,
             "M6 Phase10-C evaluation task slicing is forbidden")
    _require(args.maximum_action_tokens is None,
             "M6 Phase10-C evaluation cannot claim a training token budget")
    _require(args.replay_prefix_root is None,
             "M6 Phase10-C evaluation cannot replay a diagnostic prefix")
    _require(args.shared_prefix_manifest is None and args.state_correction_manifest is None,
             "M6 Phase10-C evaluation cannot use historical state manifests")
    identity = args.phase10c_evaluation_identity
    _require(identity in PHASE10C_EVALUATION_MODELS, "M6 Phase10-C evaluation identity drift")
    _require(args.base_model.expanduser().resolve() == PHASE10C_EVALUATION_MODELS[identity],
             "M6 Phase10-C evaluation model drift")
    if identity in ("raw35", "raw27"):
        _require(args.adapter is None and args.tensor_parallel_size == 2,
                 "M6 Phase10-C Raw35 identity drift")
    elif identity in ("sft35", "sft35_d4", "sft35_d35"):
        _require(
            args.adapter is None and args.tensor_parallel_size == 2,
            "M6 Phase10-C SFT35 identity drift",
        )
    elif identity == "sft4_d35":
        _require(
            args.adapter is not None
            and args.adapter.expanduser().resolve() == PHASE10D_S4_D35_ADAPTER
            and args.tensor_parallel_size == 1,
            "M6 Phase10-C SFT4 identity drift",
        )
    else:
        _require(
            args.adapter is not None
            and args.adapter.expanduser().resolve() == PHASE10B_STUDENT_ADAPTER
            and args.tensor_parallel_size == 1,
            "M6 Phase10-C SFT4 identity drift",
        )


def _validate_group_run_contract(
    group: Mapping[str, Any],
    *,
    task_id: str,
    group_id: str,
    k: int,
    git_sha: str,
    protocol_sha256: str,
    lineage: Mapping[str, str],
    training_updates_allowed: bool,
) -> dict[str, Any]:
    """Reject a recovered group produced by any other policy or run contract."""

    value = validate_committed_group(group, require_k=k)
    _require(value.get("task_id") == task_id, f"M6 recovered group task drift: {group_id}")
    _require(value.get("group_id") == group_id, f"M6 recovered group id drift: {group_id}")
    _require(value.get("git_sha") == git_sha, f"M6 recovered group Git drift: {group_id}")
    _require(
        value.get("protocol_sha256") == protocol_sha256,
        f"M6 recovered group protocol drift: {group_id}",
    )
    _require(
        value.get("training_updates_allowed") is training_updates_allowed,
        f"M6 recovered group training-purpose drift: {group_id}",
    )
    for field in LINEAGE_FIELDS:
        _require(
            value.get(field) == lineage[field],
            f"M6 recovered group policy lineage drift ({field}): {group_id}",
        )
    return value


async def _one_group(
    *,
    engine: RawAsyncVLLMGenerationEngine | AsyncVLLMGenerationEngine,
    loop: asyncio.AbstractEventLoop,
    loop_thread_id: int,
    task_id: str,
    goal: Mapping[str, Any],
    group_index: int,
    k: int,
    seed: int,
    iteration_index: int,
    groups_root: Path,
    episodes_root: Path,
    attempts_root: Path,
    lineage: Mapping[str, str],
    ledger: DurableTokenLedger,
    protocol_sha256: str,
    git_sha: str,
    base_url: str,
    training_updates_allowed: bool,
    max_model_turns: int,
    max_environment_steps: int,
    replay_prefix_group: Mapping[str, Any] | None = None,
    shared_prefix_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    group_id = f"g{group_index:04d}"
    _require(
        replay_prefix_group is None or shared_prefix_spec is None,
        "M6 rollout cannot combine Phase2 and Phase9 prefix sources",
    )
    if replay_prefix_group is not None:
        replay_prefix_group = validate_committed_group(replay_prefix_group, require_k=k)
        _require(replay_prefix_group["group_id"] == group_id, "M6 Phase2 replay group id drift")
        _require(replay_prefix_group["task_id"] == task_id, "M6 Phase2 replay task drift")
        for field in LINEAGE_FIELDS:
            _require(
                replay_prefix_group[field] == lineage[field],
                f"M6 Phase2 replay policy lineage drift ({field})",
            )
    destination = groups_root / f"{group_id}.json"
    if destination.is_file():
        return validate_committed_group(_load_json(destination), require_k=k)
    prior = _attempt_indices(attempts_root, group_id)
    attempt_index = max(prior, default=-1) + 1
    _require(attempt_index < MAX_INFRASTRUCTURE_ATTEMPTS, f"M6 group exhausted infrastructure attempts: {group_id}")
    attempt_start = {
        "schema_version": ATTEMPT_START_SCHEMA,
        "task_id": task_id,
        "group_id": group_id,
        "attempt_index": attempt_index,
        "K": k,
        "git_sha": git_sha,
        "protocol_sha256": protocol_sha256,
        "training_updates_allowed": training_updates_allowed,
        "policy_lineage": dict(lineage),
        "replay_prefix_group_content_sha256": (
            replay_prefix_group.get("content_sha256") if replay_prefix_group is not None else None
        ),
        "shared_prefix_manifest_content_sha256": (
            shared_prefix_spec.get("manifest_content_sha256") if shared_prefix_spec is not None else None
        ),
        "shared_prefix_source_group_content_sha256": (
            shared_prefix_spec.get("source_group_content_sha256") if shared_prefix_spec is not None else None
        ),
    }
    attempt_start["content_sha256"] = _self_hash(attempt_start)
    atomic_write_json(attempts_root / f"{group_id}.a{attempt_index}.started.json", attempt_start)

    async def one(rollout_index: int) -> dict[str, Any]:
        source_turns: list[Mapping[str, Any]] = []
        if replay_prefix_group is not None:
            source_trajectory = replay_prefix_group["trajectories"][rollout_index]
            _require(
                int(source_trajectory["rollout_index"]) == rollout_index,
                "M6 Phase2 replay rollout index drift",
            )
            source_turns = list(source_trajectory["turns"])
        elif shared_prefix_spec is not None:
            source_group = shared_prefix_spec["source_group"]
            _require(source_group["task_id"] == task_id, "M6 Phase9 shared-prefix task drift")
            source_trajectory = source_group["trajectories"][int(shared_prefix_spec["source_rollout_index"])]
            prefix_turn_count = int(shared_prefix_spec["prefix_turn_count"])
            _require(0 < prefix_turn_count < len(source_trajectory["turns"]), "M6 Phase9 prefix boundary drift")
            source_turns = list(source_trajectory["turns"][:prefix_turn_count])
        context = RolloutRequestContext(
            run_seed=seed,
            iteration_index=iteration_index,
            group_id=group_id,
            attempt_index=attempt_index,
            trajectory_id=f"{group_id}.a{attempt_index}.r{rollout_index}",
            rollout_index=rollout_index,
            shared_prefix_turns=len(source_turns) if shared_prefix_spec is not None else 0,
            sampling_attempt_index=0,
        )
        backend = ThreadsafeVLLMBackend(
            engine=engine,
            event_loop=loop,
            context=context,
            timeout_seconds=900,
            event_loop_thread_id=loop_thread_id,
            initial_turn_index=len(source_turns),
        )
        replay_backend = (
            Phase2PrefixReplayBackend(
                live_backend=backend,
                source_turns=source_turns,
                lineage=lineage,
            )
            if source_turns
            else None
        )
        environment = WebShopHTTPEnvironment(base_url=base_url, split="train", timeout_seconds=120)
        agent = QwenWebShopAgent(replay_backend or backend)
        def charge(turn: Mapping[str, Any]) -> None:
            generated_ids = list(turn.get("generated_token_ids", []))
            ledger.append(
                {
                    "schema_version": "m6_generated_turn_ledger_v1",
                    "group_id": group_id,
                    "task_id": task_id,
                    "attempt_index": attempt_index,
                    "rollout_index": rollout_index,
                    "turn_index": int(turn.get("model_turn_index", 0)),
                    "request_id": str(turn.get("request_id", "")),
                    "generated_action_tokens": len(generated_ids),
                    "generated_token_sha256": sha256_json(generated_ids),
                    "adapter_sha256": lineage["adapter_sha256"],
                }
            )

        def verify_prefix(turn: Mapping[str, Any]) -> None:
            position = int(turn.get("model_turn_index", 0)) - 1
            if position < 0 or position >= len(source_turns):
                return
            source = source_turns[position]
            _require(
                turn.get("rendered_prompt_sha256") == source.get("rendered_prompt_sha256"),
                "M6 Phase2 replay rendered prompt drift",
            )
            _require(
                list(turn.get("prompt_token_ids", [])) == list(source.get("prompt_token_ids", [])),
                "M6 Phase2 replay prompt token drift",
            )
            _require(
                list(turn.get("generated_token_ids", [])) == list(source.get("generated_token_ids", [])),
                "M6 Phase2 replay generated token drift",
            )
            _require(
                int(turn.get("sampling_seed", -1)) == int(source.get("sampling_seed", -2)),
                "M6 Phase2 replay sampling seed drift",
            )
            _require(turn.get("action") == source.get("action"), "M6 Phase2 replay action drift")
            _require(
                public_state_anchor_signature(turn["observation"])
                == source.get("pre_action_public_state_sha256"),
                "M6 Phase2 replay pre-action state drift",
            )
            _require(
                public_state_anchor_signature(turn["post_action_observation"])
                == source.get("post_action_public_state_sha256"),
                "M6 Phase2 replay post-action state drift",
            )
        try:
            episode = await asyncio.to_thread(
                run_model_episode,
                task_id,
                environment,
                agent,
                max_model_turns,
                max_environment_steps,
                charge,
                verify_prefix if source_turns else None,
            )
            if replay_backend is not None:
                _require(
                    replay_backend.consumed_prefix_turns == len(source_turns),
                    "M6 Phase2 replay terminated before consuming the audited prefix",
                )
            if shared_prefix_spec is not None:
                episode["shared_prefix_turns"] = len(source_turns)
                episode["shared_prefix_manifest_content_sha256"] = shared_prefix_spec[
                    "manifest_content_sha256"
                ]
                episode["shared_prefix_source_group_content_sha256"] = shared_prefix_spec[
                    "source_group_content_sha256"
                ]
                episode["shared_prefix_exact_replay"] = True
            if episode.get("rollout_valid") is True:
                episode = annotate_episode_with_verifier(episode, goal=goal)
            return episode
        finally:
            environment.close()

    episodes = await asyncio.gather(*(one(index) for index in range(k)))
    attempt_episode_root = attempts_root / "episodes"
    attempt_episode_root.mkdir(exist_ok=True)
    for rollout_index, episode in enumerate(episodes):
        _atomic_episode(
            attempt_episode_root / f"{group_id}.a{attempt_index}.r{rollout_index}.json",
            episode,
        )
    attempt = {
        "schema_version": ATTEMPT_SCHEMA,
        "task_id": task_id,
        "group_id": group_id,
        "attempt_index": attempt_index,
        "K": k,
        "rollout_valid": [episode.get("rollout_valid") for episode in episodes],
        "task_scores": [episode.get("task_score") for episode in episodes],
        "generated_action_tokens": sum(
            len(turn.get("generated_token_ids", []))
            for episode in episodes
            for turn in episode.get("turns", [])
        ),
    }
    attempt["content_sha256"] = _self_hash(attempt)
    atomic_write_json(attempts_root / f"{group_id}.a{attempt_index}.json", attempt)
    if not all(episode.get("rollout_valid") is True for episode in episodes):
        return None
    trajectories = []
    for rollout_index, episode in enumerate(episodes):
        trajectory_id = f"{group_id}.a{attempt_index}.r{rollout_index}"
        episode["trajectory_id"] = trajectory_id
        _atomic_episode(episodes_root / f"{trajectory_id}.json", episode)
        trajectory = trajectory_from_episode(
            episode,
            trajectory_id=trajectory_id,
            rollout_index=rollout_index,
            **lineage,
        )
        if shared_prefix_spec is not None:
            prefix_turns = int(episode["shared_prefix_turns"])
            prefix_tokens = sum(
                len(turn["generated_token_ids"]) for turn in trajectory["turns"][:prefix_turns]
            )
            trajectory.update(
                shared_prefix_turns=prefix_turns,
                shared_prefix_exact_replay=True,
                prefix_policy_loss_eligible=False,
                suffix_policy_turn_start=prefix_turns + 1,
                prefix_generated_action_tokens=prefix_tokens,
                suffix_generated_action_tokens=trajectory["generated_action_tokens"] - prefix_tokens,
                shared_prefix_manifest_content_sha256=shared_prefix_spec["manifest_content_sha256"],
                shared_prefix_source_group_content_sha256=shared_prefix_spec[
                    "source_group_content_sha256"
                ],
            )
        trajectories.append(trajectory)
    group = build_committed_group(
        task_id=task_id,
        group_id=group_id,
        attempt_index=attempt_index,
        trajectories=trajectories,
        expected_k=k,
        git_sha=git_sha,
        protocol_sha256=protocol_sha256,
        training_updates_allowed=training_updates_allowed,
        **lineage,
    )
    atomic_write_json(destination, group)
    return group


async def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_bundle = load_protocol()
    protocol = protocol_bundle["payload"]
    git_sha = _git_sha()
    _require(git_sha == protocol_bundle["git_sha"], "M6 rollout Git/protocol drift")
    split_lock = validate_split_lock(_load_json(args.split_lock))
    _require(split_lock["protocol_sha256"] == protocol_bundle["sha256"], "M6 rollout split/protocol drift")
    task_ids = list(split_lock["roles"][args.role]["task_ids"])
    roster_sha256 = None
    roster_producer_git_sha = None
    if args.task_roster is not None:
        selected, roster_sha256, roster_split_sha256, roster_protocol_sha256, roster_git_sha = _task_roster(args.task_roster)
        _require(
            roster_split_sha256 == split_lock["content_sha256"],
            "M6 task roster was frozen from a different split lock",
        )
        _require(set(selected) <= set(task_ids), "M6 task roster escapes the frozen role")
        _require(roster_protocol_sha256 == protocol_bundle["sha256"], "M6 task roster protocol drift")
        roster_producer_git_sha = validate_artifact_git_compatibility(
            artifact_name="RL curriculum",
            producer_git_sha=roster_git_sha,
            consumer_git_sha=git_sha,
            explicitly_authorized_producer_git_sha=args.task_roster_producer_git_sha,
        )
        task_ids = selected
    else:
        _require(
            args.task_roster_producer_git_sha is None,
            "M6 task-roster producer Git was provided without a task roster",
        )
    _require(args.task_offset >= 0, "M6 task offset must be non-negative")
    task_ids = task_ids[args.task_offset :]
    if args.maximum_tasks is not None:
        _require(args.maximum_tasks > 0, "M6 maximum tasks must be positive")
        task_ids = task_ids[: args.maximum_tasks]
    _require(task_ids, "M6 selected task roster is empty")
    expected_k = int(
        8
        if args.mode == "phase10c_teacher_exploration_scale"
        else 2
        if args.mode == "phase10_state_suffix_smoke"
        else protocol["mini"]["evaluation_K"]
        if args.mode in {
            "evaluation",
            "phase2_horizon_evaluation",
            "phase4_data_synthesis",
            "phase4_teacher_probe",
            "phase4_online_rl_collection",
            "phase4_tuning_evaluation",
            "phase9_shared_prefix_smoke",
            "phase10b_specialist_qualification",
            "phase10b_opd_smoke_behavior",
            "phase10c_teacher_exploration",
            "phase10c_teacher_stage_evaluation",
        }
        else protocol["rl"]["group_size"]
    )
    _require(args.k == expected_k, "M6 mode/K contract drift")
    training_updates_allowed = args.mode in {"rl_collection", "phase4_online_rl_collection"}
    _require(
        args.replay_prefix_root is None or args.mode == "phase2_horizon_evaluation",
        "M6 prefix replay is restricted to the Phase2 horizon diagnostic",
    )
    _require(
        getattr(args, "shared_prefix_manifest", None) is None or args.mode == "phase9_shared_prefix_smoke",
        "M6 shared-prefix manifest is restricted to the Phase9 smoke",
    )
    _require(
        getattr(args, "state_correction_manifest", None) is None
        or args.mode == "phase10_state_suffix_smoke",
        "M6 state-correction manifest is restricted to the Phase10 smoke",
    )
    _require(
        getattr(args, "phase10_smoke_identity", None) is None
        or args.mode == "phase10_state_suffix_smoke",
        "M6 Phase10 identity is restricted to the Phase10 smoke",
    )
    _require(
        getattr(args, "phase10b_qualification_identity", None) is None
        or args.mode == "phase10b_specialist_qualification",
        "M6 Phase10-B identity is restricted to Specialist qualification",
    )
    _require(
        getattr(args, "phase10c_evaluation_identity", None) is None
        or args.mode == "phase10c_teacher_stage_evaluation",
        "M6 Phase10-C evaluation identity is restricted to paired evaluation",
    )
    _require(
        args.tensor_parallel_size == 1
        or args.mode in {
            "phase10b_specialist_qualification",
            "phase10c_teacher_exploration",
            "phase10c_teacher_exploration_scale",
            "phase10c_teacher_stage_evaluation",
        },
        "M6 tensor parallelism is restricted to approved Phase10 teacher modes",
    )
    _require(
        not (
            getattr(args, "shared_prefix_manifest", None) is not None
            and getattr(args, "state_correction_manifest", None) is not None
        ),
        "M6 rollout cannot combine Phase9 and Phase10 manifests",
    )
    if args.mode == "raw_collection":
        _require(args.role == "mini_train", "M6 Raw collection must use mini_train")
        _require(args.adapter is None and args.task_roster is None and args.task_offset == 0, "M6 Raw collection policy/roster drift")
        _require(len(task_ids) == int(protocol["split"]["mini_train_tasks"]), "M6 Raw collection task count drift")
    elif args.mode == "evaluation":
        _require(args.role in {"mini_dev", "formal_dev"}, "M6 evaluation role drift")
        _require(args.task_roster is None and args.task_offset == 0, "M6 evaluation roster drift")
        expected_tasks = int(protocol["split"][f"{args.role}_tasks"])
        _require(len(task_ids) == expected_tasks, "M6 evaluation task count drift")
    elif args.mode == "diagnostic_evaluation":
        validate_diagnostic_evaluation_contract(args)
    elif args.mode == "phase2_horizon_evaluation":
        validate_phase2_horizon_contract(args)
    elif args.mode == "phase4_data_synthesis":
        validate_phase4_data_synthesis_contract(args)
    elif args.mode == "phase4_online_rl_collection":
        validate_phase4_online_rl_contract(args)
    elif args.mode == "phase4_tuning_evaluation":
        validate_phase4_tuning_evaluation_contract(args)
    elif args.mode == "phase4_teacher_probe":
        validate_phase4_teacher_probe_contract(args)
    elif args.mode == "phase9_shared_prefix_smoke":
        validate_phase9_shared_prefix_smoke_contract(args)
        _require(len(task_ids) == 8, "M6 Phase9 roster must contain exactly eight tasks")
    elif args.mode == "phase10_state_suffix_smoke":
        validate_phase10_state_suffix_smoke_contract(args)
        _require(len(task_ids) == 8, "M6 Phase10 smoke roster must contain exactly eight tasks")
    elif args.mode == "phase10b_specialist_qualification":
        validate_phase10b_specialist_qualification_contract(args)
        qualification_roster = _load_json(args.task_roster)
        _require(
            qualification_roster.get("identity") == args.phase10b_qualification_identity
            and qualification_roster.get("comparison_specialist") in {"S_nav", "S_match", "S_finish"}
            and args.phase10b_qualification_identity
            in {"student", qualification_roster.get("comparison_specialist")},
            "M6 Phase10-B qualification roster identity drift",
        )
        _require(
            len(task_ids) == PHASE10B_QUALIFICATION_TASK_COUNTS[args.phase10b_qualification_identity],
            "M6 Phase10-B qualification roster task count drift",
        )
    elif args.mode == "phase10b_opd_smoke_behavior":
        validate_phase10b_opd_smoke_behavior_contract(args)
        smoke_roster = _load_json(args.task_roster)
        _require(
            smoke_roster.get("schema_version") == "m6_phase10b_opd_smoke_roster_v1"
            and smoke_roster.get("phase10b_role") == "opd_smoke"
            and smoke_roster.get("identity") == "student"
            and smoke_roster.get("training_performed") is False
            and smoke_roster.get("optimizer_steps") == 0,
            "M6 Phase10-B OPD smoke roster identity drift",
        )
        _require(len(task_ids) == PHASE10B_OPD_SMOKE_TASK_COUNT,
                 "M6 Phase10-B OPD smoke roster task count drift")
    elif args.mode in PHASE10C_TEACHER_EXPLORATION_TASK_COUNTS:
        validate_phase10c_teacher_exploration_contract(args)
        exploration_roster = _load_json(args.task_roster)
        expected_role = (
            "teacher_self_exploration_scale"
            if args.mode == "phase10c_teacher_exploration_scale"
            else "teacher_self_exploration"
        )
        _require(
            exploration_roster.get("phase10c_role") == expected_role
            and exploration_roster.get("identity") == "raw_qwen3.5_35b_a3b"
            and exploration_roster.get("training_performed") is False
            and exploration_roster.get("optimizer_steps") == 0,
            "M6 Phase10-C teacher exploration roster identity drift",
        )
        _require(len(task_ids) == PHASE10C_TEACHER_EXPLORATION_TASK_COUNTS[args.mode],
                 "M6 Phase10-C teacher exploration roster task count drift")
    elif args.mode == "phase10c_teacher_stage_evaluation":
        validate_phase10c_teacher_stage_evaluation_contract(args)
        evaluation_roster = _load_json(args.task_roster)
        _require(
            evaluation_roster.get("purpose") in {
                "phase10c_teacher_stage_paired_evaluation",
                "phase10d_same_corpus_scale_paired_evaluation",
                "phase10d_same_corpus_d35_paired_evaluation",
            }
            and evaluation_roster.get("phase10c_evaluation_stage") in {
                "dev", "qualification", "same_corpus", "same_corpus_d35",
            }
            and evaluation_roster.get("task_count") == PHASE10C_EVALUATION_TASK_COUNT
            and evaluation_roster.get("K") == 4
            and evaluation_roster.get("training_performed") is False
            and evaluation_roster.get("optimizer_steps") == 0,
            "M6 Phase10-C evaluation roster identity drift",
        )
        if evaluation_roster.get("phase10c_evaluation_stage") in {"same_corpus", "same_corpus_d35"}:
            _require(
                evaluation_roster.get("rollout_seed") == args.seed,
                "M6 Phase10-D same-corpus rollout seed drift",
            )
        _require(len(task_ids) == PHASE10C_EVALUATION_TASK_COUNT,
                 "M6 Phase10-C evaluation roster task count drift")
    else:
        _require(args.role == "mini_train" and args.task_roster is not None, "M6 RL collection curriculum drift")
    _require(not training_updates_allowed or args.adapter is not None, "M6 RL collection requires an adapter")
    if args.mode == "rl_collection":
        _require(args.maximum_tasks == int(protocol["mini"]["rl_collection_groups_per_iteration"]), "M6 RL group budget drift")
        _require(args.max_model_turns == int(protocol["mini"]["rl_collection_max_model_turns"]), "M6 RL model-turn cap drift")
        _require(args.max_environment_steps == int(protocol["mini"]["rl_collection_max_environment_steps"]), "M6 RL environment-step cap drift")
        _require(args.maximum_action_tokens is not None and args.maximum_action_tokens > 0, "M6 RL action-token cap missing")
        _require(
            args.maximum_action_tokens <= int(protocol["mini"]["rl_collection_action_token_cap_per_iteration"]),
            "M6 RL iteration action-token cap drift",
        )
    elif not training_updates_allowed:
        _require(args.maximum_action_tokens is None, "M6 non-RL rollout cannot claim an RL token cap")
    output = args.output_dir.expanduser().resolve()
    groups_root, episodes_root, attempts_root = output / "groups", output / "episodes", output / "attempts"
    for path in (groups_root, episodes_root, attempts_root):
        path.mkdir(parents=True, exist_ok=True)
    ledger = DurableTokenLedger(output / "generated_turn_ledger.jsonl")
    goals = _load_json(args.goals)
    _require(sha256_json(goals) == split_lock["goals_canonical_sha256"], "M6 rollout goals/split drift")
    goal_map = {f"webshop_goal_{int(item['goal_index']):05d}": item for item in goals}
    _require(all(task_id in goal_map for task_id in task_ids), "M6 rollout goal roster is incomplete")
    replay_prefix_groups: dict[str, dict[str, Any]] = {}
    replay_prefix_source: dict[str, Any] | None = None
    if args.replay_prefix_root is not None:
        replay_root = args.replay_prefix_root.expanduser().resolve()
        replay_report = _load_json(replay_root / "collection_report.json")
        _require(
            replay_report.get("content_sha256") == _self_hash(replay_report),
            "M6 Phase2 replay source report self-hash drift",
        )
        _require(
            replay_report.get("mode") == "phase2_horizon_evaluation"
            and replay_report.get("K") == args.k
            and replay_report.get("task_count") == len(task_ids),
            "M6 Phase2 replay source shape drift",
        )
        _require(
            replay_report.get("task_order_sha256") == sha256_json(task_ids)
            and replay_report.get("task_roster_content_sha256") == roster_sha256
            and replay_report.get("split_lock_content_sha256") == split_lock["content_sha256"]
            and replay_report.get("protocol_sha256") == protocol_bundle["sha256"],
            "M6 Phase2 replay source contract drift",
        )
        for index, task_id in enumerate(task_ids):
            value = validate_committed_group(
                _load_json(replay_root / "groups" / f"g{index:04d}.json"),
                require_k=args.k,
            )
            _require(value["task_id"] == task_id, "M6 Phase2 replay source task order drift")
            replay_prefix_groups[value["group_id"]] = value
        _require(
            replay_report.get("group_content_sha256")
            == [replay_prefix_groups[f"g{index:04d}"]["content_sha256"] for index in range(len(task_ids))],
            "M6 Phase2 replay source group/report drift",
        )
        replay_prefix_source = {
            "root": str(replay_root),
            "collection_report_content_sha256": replay_report["content_sha256"],
            "producer_git_sha": replay_report["git_sha"],
            "group_content_sha256": list(replay_report["group_content_sha256"]),
        }
    shared_prefix_specs: dict[str, dict[str, Any]] = {}
    shared_prefix_source: dict[str, Any] | None = None
    prefix_manifest_arg = (
        getattr(args, "shared_prefix_manifest", None)
        or getattr(args, "state_correction_manifest", None)
    )
    phase10_prefix = getattr(args, "state_correction_manifest", None) is not None
    if prefix_manifest_arg is not None:
        manifest_path = prefix_manifest_arg.expanduser().resolve()
        _require(
            not ({"promotion", "holdout"} & {part.casefold() for part in manifest_path.parts}),
            "M6 prefix manifest touches promotion/holdout",
        )
        manifest = _load_json(manifest_path)
        _require(manifest.get("content_sha256") == _self_hash(manifest), "M6 prefix manifest self-hash drift")
        expected_manifest_schema = (
            "m6_phase10_state_suffix_smoke_manifest_v1"
            if phase10_prefix
            else "m6_phase9_shared_prefix_manifest_v1"
        )
        _require(
            manifest.get("schema_version") == expected_manifest_schema
            and manifest.get("complete") is True
            and manifest.get("development_only") is True
            and manifest.get("training_performed") is False
            and manifest.get("optimizer_steps") == 0
            and manifest.get("public_fields_only") is True
            and manifest.get("target_asin_or_hidden_answer_used") is False
            and manifest.get("post_action_internal_state_used") is False,
            "M6 prefix manifest contract drift",
        )
        if phase10_prefix:
            _require(
                manifest.get("task_score_used_for_selector") is False
                and manifest.get("correction_selector")
                == "last_public_prebuy_state_for_all_nonstrict_purchase_v1",
                "M6 Phase10 public correction selector drift",
            )
        _require(
            manifest.get("producer_git_sha") == git_sha
            and manifest.get("protocol_sha256") == protocol_bundle["sha256"]
            and manifest.get("split_lock_content_sha256") == split_lock["content_sha256"]
            and manifest.get("task_roster_content_sha256") == roster_sha256
            and manifest.get("task_order_sha256") == sha256_json(task_ids)
            and manifest.get("K") == args.k
            and manifest.get("task_count") == len(task_ids),
            "M6 prefix manifest identity drift",
        )
        manifest_tasks = manifest.get("tasks")
        _require(
            isinstance(manifest_tasks, list)
            and [row.get("task_id") for row in manifest_tasks] == task_ids,
            "M6 prefix manifest task order drift",
        )
        source_k = 4 if phase10_prefix else args.k
        for row in manifest_tasks:
            root = Path(str(row["source_collection_root"])).expanduser().resolve()
            _require(
                not ({"promotion", "holdout"} & {part.casefold() for part in root.parts}),
                "M6 prefix source touches promotion/holdout",
            )
            source_report = _load_json(root / "collection_report.json")
            _require(
                source_report.get("content_sha256") == _self_hash(source_report)
                and source_report.get("content_sha256")
                == row["source_collection_report_content_sha256"]
                and source_report.get("policy_lineage") == row["source_policy_lineage"],
                "M6 prefix source report binding drift",
            )
            if phase10_prefix:
                _require(
                    source_report.get("mode") == "phase4_data_synthesis"
                    and source_report.get("K") == 4,
                    "M6 Phase10 historical source shape drift",
                )
            group_path = root / "groups" / f"{row['source_group_id']}.json"
            source_group = validate_committed_group(_load_json(group_path), require_k=source_k)
            _require(
                source_group["content_sha256"] == row["source_group_content_sha256"]
                and source_group["content_sha256"] in source_report["group_content_sha256"]
                and source_group["task_id"] == row["task_id"],
                "M6 prefix source group binding drift",
            )
            rollout_index = int(row["source_rollout_index"])
            _require(0 <= rollout_index < source_k, "M6 prefix source rollout index drift")
            source_trajectory = source_group["trajectories"][rollout_index]
            prefix_turn_count = int(row["prefix_turn_count"])
            _require(0 < prefix_turn_count < len(source_trajectory["turns"]), "M6 prefix boundary drift")
            prefix_turns = source_trajectory["turns"][:prefix_turn_count]
            commands = [str((turn.get("action") or {}).get("command", "")).strip() for turn in prefix_turns]
            token_evidence = [
                {
                    "prompt_token_ids": list(turn["prompt_token_ids"]),
                    "generated_token_ids": list(turn["generated_token_ids"]),
                    "behavior_logprobs": list(turn["behavior_logprobs"]),
                    "sampling_logprobs": list(turn["sampling_logprobs"]),
                }
                for turn in prefix_turns
            ]
            prefix_evidence_ok = (
                sha256_json(commands) == row["prefix_action_sequence_sha256"]
                and sha256_json(token_evidence) == row["prefix_token_evidence_sha256"]
            )
            if phase10_prefix:
                _require(
                    prefix_evidence_ok
                    and source_trajectory["turns"][prefix_turn_count]["pre_action_public_state_sha256"]
                    == row["correction_public_state_sha256"]
                    and _normalized_command(
                        source_trajectory["turns"][prefix_turn_count]["action"]["command"]
                    )
                    == "click[buy now]",
                    "M6 Phase10 source correction evidence drift",
                )
            else:
                _require(
                    prefix_evidence_ok
                    and re.fullmatch(r"click\[B[0-9A-Z]{9}\]", commands[-1], re.IGNORECASE)
                    is not None,
                    "M6 Phase9 source prefix evidence drift",
                )
            spec = dict(row)
            spec["source_group"] = source_group
            spec["manifest_content_sha256"] = manifest["content_sha256"]
            shared_prefix_specs[str(row["task_id"])] = spec
        _require(set(shared_prefix_specs) == set(task_ids), "M6 prefix source roster incomplete")
        shared_prefix_source = {
            "manifest": str(manifest_path),
            "manifest_content_sha256": manifest["content_sha256"],
            "producer_git_sha": manifest["producer_git_sha"],
            "policy_lineage": dict(manifest["source_policy_lineage"] if phase10_prefix else manifest["policy_lineage"]),
            "future_policy_loss_contract": dict(manifest["future_policy_loss_contract"]),
            "manifest_schema_version": expected_manifest_schema,
            "phase10_smoke_identity": args.phase10_smoke_identity if phase10_prefix else None,
        }
    # Reconstruct already charged budget before allocating GPU memory.  A
    # timeout after generation but before group commit must not receive a new
    # budget simply because this process restarted.
    if args.maximum_action_tokens is not None:
        _require(
            ledger.audit()["generated_action_tokens"] <= args.maximum_action_tokens,
            "M6 persisted token ledger already exceeds the action-token cap",
        )
    engine, lineage = await _create_identity(
        base_model=args.base_model.expanduser().resolve(),
        base_model_manifest=args.base_model_manifest.expanduser().resolve(),
        adapter=args.adapter,
        output=output,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if replay_prefix_source is not None:
        _require(
            replay_report.get("policy_lineage") == dict(lineage),
            "M6 Phase2 replay source SFT policy drift",
        )
    if shared_prefix_source is not None:
        if phase10_prefix:
            if args.phase10_smoke_identity == "student":
                _require(
                    shared_prefix_source["policy_lineage"] == dict(lineage),
                    "M6 Phase10 student/source pi_0 lineage drift",
                )
            else:
                _require(
                    shared_prefix_source["policy_lineage"] != dict(lineage),
                    "M6 Phase10 teacher unexpectedly reuses the student lineage",
                )
        else:
            _require(
                shared_prefix_source["policy_lineage"] == dict(lineage),
                "M6 Phase9 shared-prefix SFT policy drift",
            )
    loop = asyncio.get_running_loop()
    loop_thread_id = threading.get_ident()
    try:
        invocation = {
            "schema_version": "m6_rollout_invocation_v1",
            "development_only": True,
            "formal_training": False,
            "mode": args.mode,
            "role": args.role,
            "K": args.k,
            "task_count": len(task_ids),
            "task_order_sha256": sha256_json(task_ids),
            "task_roster_content_sha256": roster_sha256,
            "task_roster_producer_git_sha": roster_producer_git_sha,
            "task_roster_consumer_git_sha": git_sha if roster_sha256 is not None else None,
            "task_offset": args.task_offset,
            "seed": args.seed,
            "iteration_index": args.iteration_index,
            "max_model_turns": args.max_model_turns,
            "max_environment_steps": args.max_environment_steps,
            "maximum_action_tokens": args.maximum_action_tokens,
            "training_updates_allowed": training_updates_allowed,
            "protocol_sha256": protocol_bundle["sha256"],
            "git_sha": git_sha,
            "split_lock_content_sha256": split_lock["content_sha256"],
            "base_model": str(args.base_model.expanduser().resolve()),
            "base_model_manifest": str(args.base_model_manifest.expanduser().resolve()),
            "base_model_manifest_sha256": lineage["adapter_sha256"],
            "adapter": str(args.adapter.expanduser().resolve()) if args.adapter else None,
            "policy_lineage": dict(lineage),
            "replay_prefix_source": replay_prefix_source,
            "shared_prefix_source": shared_prefix_source,
            "phase10_smoke_identity": getattr(args, "phase10_smoke_identity", None),
            "phase10b_qualification_identity": getattr(args, "phase10b_qualification_identity", None),
            "phase10c_evaluation_identity": getattr(args, "phase10c_evaluation_identity", None),
            "tensor_parallel_size": args.tensor_parallel_size,
        }
        invocation["content_sha256"] = _self_hash(invocation)
        invocation_path = output / "invocation.json"
        if invocation_path.is_file():
            _require(_load_json(invocation_path) == invocation, "M6 rollout invocation changed across recovery")
        else:
            atomic_write_json(invocation_path, invocation)

        # A 24-hour successor may only reuse groups from this exact policy,
        # split, task order, protocol and purpose.  This prevents a partially
        # completed directory from silently mixing two adapter generations.
        for index, task_id in enumerate(task_ids):
            existing = groups_root / f"g{index:04d}.json"
            if existing.is_file():
                _validate_group_run_contract(
                    _load_json(existing),
                    task_id=task_id,
                    group_id=f"g{index:04d}",
                    k=args.k,
                    git_sha=git_sha,
                    protocol_sha256=protocol_bundle["sha256"],
                    lineage=lineage,
                    training_updates_allowed=training_updates_allowed,
                )
        while True:
            pending = [
                (index, task_id)
                for index, task_id in enumerate(task_ids)
                if not (groups_root / f"g{index:04d}.json").is_file()
            ]
            if not pending:
                break
            wave = pending[: args.concurrent_groups]
            if args.maximum_action_tokens is not None:
                # Reserve the mathematical maximum before starting a complete
                # K8 attempt.  This counts every generated token, including
                # infrastructure-invalid attempts, and therefore cannot
                # overshoot the frozen cap after generation has already run.
                worst_case_attempt = len(wave) * args.k * args.max_model_turns * MAX_NEW_TOKENS
                _require(
                    ledger.audit()["generated_action_tokens"] + worst_case_attempt
                    <= args.maximum_action_tokens,
                    "M6 action-token budget cannot reserve another complete rollout attempt",
                )
            results = await asyncio.gather(
                *(
                    _one_group(
                        engine=engine,
                        loop=loop,
                        loop_thread_id=loop_thread_id,
                        task_id=task_id,
                        goal=goal_map[task_id],
                        group_index=index,
                        k=args.k,
                        seed=args.seed,
                        iteration_index=args.iteration_index,
                        groups_root=groups_root,
                        episodes_root=episodes_root,
                        attempts_root=attempts_root,
                        lineage=lineage,
                        ledger=ledger,
                        protocol_sha256=protocol_bundle["sha256"],
                        git_sha=git_sha,
                        base_url=args.base_url,
                        training_updates_allowed=training_updates_allowed,
                        max_model_turns=args.max_model_turns,
                        max_environment_steps=args.max_environment_steps,
                        replay_prefix_group=replay_prefix_groups.get(f"g{index:04d}"),
                        shared_prefix_spec=shared_prefix_specs.get(task_id),
                    )
                    for index, task_id in wave
                )
            )
            # An invalid infrastructure attempt is retried on the next wave;
            # algorithm failures are valid completed groups and never retried.
            _require(any(item is not None for item in results) or all(
                max(_attempt_indices(attempts_root, f"g{index:04d}"), default=-1) + 1
                < MAX_INFRASTRUCTURE_ATTEMPTS
                for index, _ in wave
            ), "M6 rollout wave exhausted infrastructure retries")
    finally:
        engine.shutdown()
    groups = [
        _validate_group_run_contract(
            _load_json(groups_root / f"g{index:04d}.json"),
            task_id=task_id,
            group_id=f"g{index:04d}",
            k=args.k,
            git_sha=git_sha,
            protocol_sha256=protocol_bundle["sha256"],
            lineage=lineage,
            training_updates_allowed=training_updates_allowed,
        )
        for index, task_id in enumerate(task_ids)
    ]
    trajectories = [trajectory for group in groups for trajectory in group["trajectories"]]
    ledger_audit = ledger.audit()
    _require(
        ledger_audit["generated_action_tokens"] >= sum(group["generated_action_tokens"] for group in groups),
        "M6 token ledger excludes committed group tokens",
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "complete": True,
        "development_only": True,
        "mode": args.mode,
        "role": args.role,
        "task_count": len(groups),
        "trajectory_count": len(trajectories),
        "K": args.k,
        "strict_success_task_count": len({group["task_id"] for group in groups if any(item["success"] for item in group["trajectories"])}),
        "strict_success_trajectory_count": sum(item["success"] for item in trajectories),
        "mixed_strict_reward_group_count": sum(
            len({item["binary_reward"] for item in group["trajectories"]}) > 1 for group in groups
        ),
        "generated_action_tokens": sum(group["generated_action_tokens"] for group in groups),
        "all_attempt_generated_action_tokens": ledger_audit["generated_action_tokens"],
        "maximum_action_tokens": args.maximum_action_tokens,
        "training_updates_allowed": training_updates_allowed,
        "action_token_budget_respected": (
            args.maximum_action_tokens is None
            or ledger_audit["generated_action_tokens"] <= args.maximum_action_tokens
        ),
        "token_ledger": ledger_audit,
        "git_sha": git_sha,
        "protocol_sha256": protocol_bundle["sha256"],
        "split_lock_content_sha256": split_lock["content_sha256"],
        "task_order_sha256": sha256_json(task_ids),
        "task_roster_content_sha256": roster_sha256,
        "task_roster_producer_git_sha": roster_producer_git_sha,
        "task_roster_consumer_git_sha": git_sha if roster_sha256 is not None else None,
        "invocation_content_sha256": invocation["content_sha256"],
        "policy_lineage": dict(lineage),
        "group_content_sha256": [group["content_sha256"] for group in groups],
        "replay_prefix_source": replay_prefix_source,
        "shared_prefix_source": shared_prefix_source,
        "phase10_smoke_identity": getattr(args, "phase10_smoke_identity", None),
        "phase10b_qualification_identity": getattr(args, "phase10b_qualification_identity", None),
        "phase10c_evaluation_identity": getattr(args, "phase10c_evaluation_identity", None),
        "tensor_parallel_size": args.tensor_parallel_size,
        "exact_shared_prefix_replay_task_count": (
            sum(
                all(
                    trajectory.get("shared_prefix_exact_replay") is True
                    and trajectory.get("prefix_policy_loss_eligible") is False
                    and int(trajectory.get("shared_prefix_turns", 0)) > 0
                    for trajectory in group["trajectories"]
                )
                for group in groups
            )
            if shared_prefix_source is not None
            else 0
        ),
        "future_policy_loss_scope": "suffix_tokens_only" if shared_prefix_source is not None else None,
        "elapsed_seconds": time.time() - args.started_at,
    }
    report["content_sha256"] = _self_hash(report)
    atomic_write_json(output / "collection_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "raw_collection",
            "evaluation",
            "diagnostic_evaluation",
            "phase2_horizon_evaluation",
            "phase4_data_synthesis",
            "phase4_teacher_probe",
            "phase4_online_rl_collection",
            "phase4_tuning_evaluation",
            "phase9_shared_prefix_smoke",
            "phase10_state_suffix_smoke",
            "phase10b_specialist_qualification",
            "phase10b_opd_smoke_behavior",
            "phase10c_teacher_exploration",
            "phase10c_teacher_exploration_scale",
            "phase10c_teacher_stage_evaluation",
            "rl_collection",
        ),
        required=True,
    )
    parser.add_argument("--role", choices=("mini_train", "mini_dev", "formal_dev", "train"), required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--base-model-manifest", type=Path, default=BASE_MODEL_MANIFEST_PATH)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--k", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--iteration-index", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--maximum-tasks", type=int)
    parser.add_argument("--task-roster", type=Path)
    parser.add_argument("--task-roster-producer-git-sha")
    parser.add_argument("--replay-prefix-root", type=Path)
    parser.add_argument("--shared-prefix-manifest", type=Path)
    parser.add_argument("--state-correction-manifest", type=Path)
    parser.add_argument("--phase10-smoke-identity", choices=("student", "teacher"))
    parser.add_argument("--phase10b-qualification-identity", choices=tuple(PHASE10B_QUALIFICATION_MODELS))
    parser.add_argument("--phase10c-evaluation-identity", choices=tuple(PHASE10C_EVALUATION_MODELS))
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--max-model-turns", type=int, default=18)
    parser.add_argument("--max-environment-steps", type=int, default=15)
    parser.add_argument("--maximum-action-tokens", type=int)
    parser.add_argument("--concurrent-groups", type=int, default=4)
    parser.add_argument("--base-url", default="http://127.0.0.1:44151")
    args = parser.parse_args()
    _require(1 <= args.concurrent_groups <= 4, "M6 group concurrency must be in [1,4]")
    _require(args.max_model_turns > 0 and args.max_environment_steps > 0, "M6 turn caps must be positive")
    args.started_at = time.time()
    return args


def main() -> None:
    report = asyncio.run(run(parse_args()))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
