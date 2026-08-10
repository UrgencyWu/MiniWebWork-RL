"""Fail-closed protocol helpers for the focused M5 WebShop study."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "data" / "m5_webshop_study_v1.json"
UPSTREAM_LOCK_PATH = PROJECT_ROOT / "data" / "m5_webshop_upstream_lock_v1.json"
SPLIT_EXCLUSIONS_PATH = PROJECT_ROOT / "data" / "m5_webshop_split_exclusions_v1.json"

STUDY_ID = "m5_webshop_credit_assignment_v1"
SCHEMA_VERSION = "m5_webshop_study_v1"
LOCK_SCHEMA_VERSION = "m5_webshop_upstream_lock_v1"
SPLIT_EXCLUSIONS_SCHEMA_VERSION = "m5_webshop_split_exclusions_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def repository_git_sha() -> str:
    """Return the exact repository revision bound into every M5 artifact."""

    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(GIT_REVISION_RE.fullmatch(result) is not None, "invalid M5 repository Git SHA")
    return result


def content_tree_audit(
    root: Path,
    prefix: str,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Hash a source subtree by relative path, size and per-file SHA256."""

    source = Path(root).expanduser().resolve()
    tree_root = source / prefix if prefix else source
    _require(tree_root.is_dir(), f"content tree is missing: {prefix}")
    files = []
    for path in tree_root.rglob("*"):
        relative = path.relative_to(source).as_posix()
        if any(relative == blocked or relative.startswith(f"{blocked}/") for blocked in excluded_prefixes):
            continue
        _require(not path.is_symlink(), f"content tree contains a symlink: {path}")
        if not path.is_file():
            continue
        files.append(path)
    files.sort()
    _require(bool(files), f"content tree is empty: {prefix}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(source).as_posix()
        size = path.stat().st_size
        file_sha = sha256_file(path)
        digest.update(f"{relative}\0{size}\0{file_sha}\n".encode("utf-8"))
        total_bytes += size
    return {
        "prefix": prefix or ".",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def validate_upstream_lock(payload: Mapping[str, Any]) -> dict[str, Any]:
    lock = dict(payload)
    _require(lock.get("schema_version") == LOCK_SCHEMA_VERSION, "upstream lock schema drift")
    source = lock.get("source")
    _require(isinstance(source, Mapping), "upstream lock source is missing")
    _require(GIT_REVISION_RE.fullmatch(str(source.get("revision", ""))) is not None, "invalid data revision")
    _require(source.get("runtime_root") == "webshop_full", "unexpected upstream runtime root")
    files = lock.get("runtime_files")
    _require(isinstance(files, list) and files, "upstream runtime file lock is empty")
    paths: set[str] = set()
    total = 0
    for item in files:
        _require(isinstance(item, Mapping), "invalid upstream file entry")
        path = str(item.get("path", ""))
        _require(path and not path.startswith("/") and ".." not in Path(path).parts, "unsafe upstream path")
        _require(path not in paths, f"duplicate upstream path: {path}")
        paths.add(path)
        _require(SHA256_RE.fullmatch(str(item.get("sha256", ""))) is not None, f"invalid SHA256: {path}")
        size = item.get("size")
        _require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, f"invalid size: {path}")
        total += size
    required = {"goals.json", "products.sqlite", "train.parquet", "test.parquet", "meta.json", "stats.json", "lucene_index/segments_1"}
    _require(required <= paths, "upstream runtime lock lacks required files")
    _require(total == lock.get("total_runtime_bytes"), "upstream runtime byte total drift")
    return lock


def load_upstream_lock(path: Path = UPSTREAM_LOCK_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = validate_upstream_lock(_json(resolved))
    return {"path": str(resolved), "sha256": sha256_file(resolved), "payload": payload}


def _validate_split(
    name: str,
    value: Mapping[str, Any],
    *,
    start: int,
    stop: int,
    eligible_count: int,
    may_update: bool,
) -> None:
    _require(value.get("start_inclusive") == start, f"{name} split start drift")
    _require(value.get("stop_exclusive") == stop, f"{name} split stop drift")
    _require(value.get("raw_count") == stop - start, f"{name} raw split count drift")
    _require(value.get("eligible_count") == eligible_count, f"{name} eligible split count drift")
    _require(value.get("may_update_model") is may_update, f"{name} split role drift")


def _compact_index_sha256(indices: list[int]) -> str:
    return hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_split_exclusions(payload: Mapping[str, Any]) -> dict[str, Any]:
    exclusions = dict(payload)
    _require(
        exclusions.get("schema_version") == SPLIT_EXCLUSIONS_SCHEMA_VERSION,
        "M5 split exclusions schema drift",
    )
    goals_sha = next(
        item["sha256"]
        for item in load_upstream_lock()["payload"]["runtime_files"]
        if item["path"] == "goals.json"
    )
    _require(exclusions.get("source_goals_sha256") == goals_sha, "M5 split exclusions source drift")
    _require(
        exclusions.get("normalization") == "unicode_preserving_lowercase_whitespace_collapse_v1",
        "M5 instruction normalization drift",
    )
    expected_bounds = {"test": (0, 500), "dev": (500, 1000), "train": (1000, 12087)}
    expected_counts = {"test": 500, "dev": 499, "train": 10885}
    excluded = exclusions.get("excluded_goal_indices")
    _require(isinstance(excluded, Mapping), "M5 split exclusions are missing")
    _require(excluded.get("test") == [], "the frozen M5 test roster may not be filtered")
    for split, (start, stop) in expected_bounds.items():
        values = excluded.get(split)
        _require(isinstance(values, list), f"M5 {split} exclusions are missing")
        _require(values == sorted(set(values)), f"M5 {split} exclusions are not unique and sorted")
        _require(
            all(isinstance(index, int) and not isinstance(index, bool) and start <= index < stop for index in values),
            f"M5 {split} exclusion is outside its role",
        )
        eligible = [index for index in range(start, stop) if index not in set(values)]
        _require(len(eligible) == expected_counts[split], f"M5 {split} eligible count drift")
        _require(exclusions.get("eligible_counts", {}).get(split) == len(eligible), f"M5 {split} count lock drift")
        _require(
            exclusions.get("eligible_index_sha256", {}).get(split) == _compact_index_sha256(eligible),
            f"M5 {split} eligible roster hash drift",
        )
    intersections = exclusions.get("raw_cross_role_normalized_instruction_intersections")
    _require(intersections == {"test_dev": 1, "test_train": 15, "dev_train": 12}, "M5 raw overlap audit drift")
    return exclusions


def load_split_exclusions(path: Path = SPLIT_EXCLUSIONS_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = validate_split_exclusions(_json(resolved))
    return {"path": str(resolved), "sha256": sha256_file(resolved), "payload": payload}


def validate_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol = dict(payload)
    _require(protocol.get("schema_version") == SCHEMA_VERSION, "M5 protocol schema drift")
    _require(protocol.get("study_id") == STUDY_ID, "M5 study id drift")
    _require(protocol.get("status") == "preflight_only", "M5 protocol status must remain preflight_only")
    _require(protocol.get("formal_submission_allowed") is False, "formal M5 submission was enabled inside the study protocol")
    scope = protocol.get("scope")
    _require(isinstance(scope, Mapping), "M5 scope is missing")
    _require(scope.get("online_methods") == ["multi_turn_grpo", "anchor_gigpo"], "M5 method matrix drift")
    _require(scope.get("online_seeds") == [20260801, 20260802, 20260803], "M5 seed matrix drift")
    _require(scope.get("formal_models") == 8, "M5 formal model count drift")
    sources = protocol.get("upstream_sources")
    _require(isinstance(sources, Mapping), "M5 upstream sources are missing")
    expected_source_licenses = {
        "webshop": "MIT",
        "agent_r1_code": "MIT",
        "agent_r1_data": "inherits original benchmark terms; research use with attribution; artifacts are not redistributed",
    }
    for name in ("webshop", "agent_r1_code", "agent_r1_data"):
        source = sources.get(name)
        _require(isinstance(source, Mapping), f"missing upstream source: {name}")
        _require(GIT_REVISION_RE.fullmatch(str(source.get("revision", ""))) is not None, f"invalid source revision: {name}")
        _require(source.get("license") == expected_source_licenses[name], f"source license drift: {name}")
    agent_r1_source = sources["agent_r1_code"]
    _require(
        agent_r1_source.get("archive_url")
        == "https://codeload.github.com/AgentR1/Agent-R1/tar.gz/b124aa46534cbf2fb8bc8af11405774984c42ac7",
        "Agent-R1 archive URL drift",
    )
    _require(
        agent_r1_source.get("archive_sha256")
        == "07e6a35a159e7ed148d1e4b2b47d5e0158e3b8f60a71e5626f911477dfc7d57b",
        "Agent-R1 archive SHA256 drift",
    )
    _require(agent_r1_source.get("archive_size") == 1628704, "Agent-R1 archive size drift")
    _require(agent_r1_source.get("archive_member_count") == 226, "Agent-R1 archive roster drift")
    _require(
        agent_r1_source.get("content_tree_contract")
        == "SHA256 over sorted repository-relative regular-file records: UTF-8 path, NUL, decimal byte size, NUL, lowercase file SHA256, LF; symlinks forbidden; excluded control paths omitted before traversal checks",
        "Agent-R1 content-tree algorithm drift",
    )
    _require(
        agent_r1_source.get("source_content_tree_sha256")
        == "f45b0e09e4a500c3c9915159c7e395952a16d12378924c5f99ce8d42d55ecd9a",
        "Agent-R1 source content-tree drift",
    )
    _require(agent_r1_source.get("source_content_tree_file_count") == 178, "Agent-R1 source file-count drift")
    _require(agent_r1_source.get("source_content_tree_bytes") == 2189338, "Agent-R1 source byte-count drift")
    _require(
        agent_r1_source.get("webshop_content_tree_sha256")
        == "bf79abafad937aa6da0a5cbec69766f3bd1bd5420407e38a58e17d1b36b51c1f",
        "Agent-R1 WebShop content-tree drift",
    )
    dataset = protocol.get("dataset")
    _require(isinstance(dataset, Mapping), "M5 dataset contract is missing")
    _require(dataset.get("products") == 1181430 and dataset.get("goals") == 12087, "WebShop count drift")
    splits = dataset.get("split_contract")
    _require(isinstance(splits, Mapping), "M5 split contract is missing")
    _validate_split("test", splits["test"], start=0, stop=500, eligible_count=500, may_update=False)
    _validate_split("dev", splits["dev"], start=500, stop=1000, eligible_count=499, may_update=False)
    _validate_split("train", splits["train"], start=1000, stop=12087, eligible_count=10885, may_update=True)
    dedup = dataset.get("dedup_contract")
    _require(isinstance(dedup, Mapping), "M5 dedup contract is missing")
    _require(dedup.get("lock_path") == "data/m5_webshop_split_exclusions_v1.json", "M5 dedup lock drift")
    load_split_exclusions(PROJECT_ROOT / dedup["lock_path"])
    _require(
        dataset.get("public_action_normalization")
        == "normalize whitespace, preserve first occurrence order, and stably deduplicate byte-identical executable commands before enforcing the 256-action bound; malformed non-string or empty actions remain fail-closed",
        "M5 public-action normalization drift",
    )
    sft = protocol.get("sft")
    _require(isinstance(sft, Mapping), "M5 SFT contract is missing")
    _require(sft.get("external_trajectory_data") is False, "external trajectory data entered formal M5 SFT")
    _require(sft.get("train_task_count") == 4000 and sft.get("dev_task_count") == 400, "M5 SFT roster drift")
    _require(
        sft.get("maximum_teacher_query_characters") == 200 and sft.get("maximum_oracle_turns") == 15,
        "M5 SFT teacher bound drift",
    )
    _require(sft.get("chat_template_kwargs") == {"enable_thinking": False}, "M5 SFT chat template drift")
    _require(sft.get("minimum_epochs") == 1 and sft.get("maximum_epochs") == 2, "M5 SFT epoch bound drift")
    _require(sft.get("minimum_effective_completion_label_token_exposure") == 250000, "M5 SFT token exposure drift")
    _require(sft.get("maximum_zero_label_fraction") == 0.0, "M5 SFT zero-label gate drift")
    online = protocol.get("online")
    _require(isinstance(online, Mapping), "M5 online contract is missing")
    _require(online.get("group_size") == 4, "M5 group size drift")
    _require(online.get("generated_action_token_cap_per_run") == 500000, "M5 action-token budget drift")
    _require(
        online.get("initial_parallel_lanes") == 32
        and online.get("candidate_parallel_lanes") == [32, 64]
        and online.get("selected_parallel_lanes") == 32,
        "M5 online lane selection drift",
    )
    _require(
        online.get("credit_formula_version") == "webshop_public_state_macro_micro_v1",
        "M5 credit formula drift",
    )
    _require(
        online.get("anchor_contract")
        == {
            "state_source": "exact task-scoped public MDP state before action: schema, task id, instruction, page type, prompt-visible observation text, truncation flag, bounded available actions and terminal flag",
            "excluded_from_grouping": [
                "prompt tokens",
                "step index",
                "recent action history",
                "episode id",
                "runtime URL",
                "oracle target",
                "verifier output",
                "unknown fields",
            ],
            "policy_context_evidence": "exact prompt-token SHA256 is recorded separately and binds behavior logprobs but never changes the state group",
            "group_scope": "same frozen task and same atomic K4 rollout group only",
        },
        "M5 public-state anchor contract drift",
    )
    _require(
        online.get("credit_parameters")
        == {
            "advantage_epsilon": 1e-6,
            "micro_return_gamma": 0.95,
            "micro_advantage_weight": 1.0,
            "anchor_visit_policy": "first_visit_per_trajectory",
            "advantage_normalization": "population standardization within the same task K4 group; exact zero when standard deviation is at most epsilon",
            "loss_normalization": "token mean within turn; turn mean within trajectory; trajectory mean within K4 group",
        },
        "M5 credit parameters drift",
    )
    gates = protocol.get("preflight_gates")
    _require(isinstance(gates, Mapping), "M5 preflight gates are missing")
    _require(
        gates.get("artifact_lineage")
        == "every M5 runtime, data, environment, health, service-stress, SFT record, corpus and token-audit artifact embeds the exact clean 40-character repository Git SHA in addition to the protocol SHA256",
        "M5 artifact-lineage contract drift",
    )
    _require(
        gates.get("throughput")
        == "benchmark 32 and 64 lanes against one shared 24-CPU service with 8 and 16 process-serialized workers; select the faster stable setting; HTTP 5xx fraction must be zero; generation-phase median GPU utilization >= 0.60 and learner-phase median GPU utilization >= 0.80; no OOM",
        "M5 throughput gate drift",
    )
    evaluation = protocol.get("evaluation")
    _require(isinstance(evaluation, Mapping), "M5 evaluation contract is missing")
    _require(evaluation.get("task_count") == 500 and evaluation.get("rollouts_per_task") == 4, "M5 frozen evaluation drift")
    _require(evaluation.get("models") == 8 and evaluation.get("total_frozen_trajectories") == 16000, "M5 frozen cost drift")
    slurm = protocol.get("slurm")
    _require(isinstance(slurm, Mapping), "M5 Slurm contract is missing")
    _require(slurm.get("wall_time_per_job") == "24:00:00", "M5 wall-time drift")
    _require(
        slurm.get("cpu_regression") == {"gpus": 0, "cpus": 2, "memory_gib": 8},
        "M5 CPU regression resource drift",
    )
    _require(
        slurm.get("server_setup") == {"gpus": 0, "cpus": 2, "memory_gib": 20},
        "M5 server setup resource drift",
    )
    _require(
        slurm.get("training_runtime_setup") == {"gpus": 0, "cpus": 2, "memory_gib": 8},
        "M5 training runtime setup resource drift",
    )
    _require(
        slurm.get("sft_corpus") == {"gpus": 0, "cpus": 16, "memory_gib": 32, "workers": 16},
        "M5 SFT corpus resource drift",
    )
    _require(
        slurm.get("service_health") == {"gpus": 0, "cpus": 2, "memory_gib": 8},
        "M5 service health resource drift",
    )
    _require(
        slurm.get("service_stress") == {"gpus": 0, "cpus": 8, "memory_gib": 16, "episodes_per_lane": 4},
        "M5 service stress resource drift",
    )
    _require(
        slurm.get("sft") == {"gpus": 1, "cpus": 8, "memory_gib": 48},
        "M5 SFT resource drift",
    )
    _require(
        slurm.get("online_per_run") == {"gpus": 1, "cpus": 8, "memory_gib": 32},
        "M5 online resource drift",
    )
    _require(
        slurm.get("evaluation_per_run") == {"gpus": 1, "cpus": 6, "memory_gib": 24},
        "M5 evaluation resource drift",
    )
    _require(slurm.get("maximum_parallel_online_runs") == 6, "M5 online parallelism drift")
    _require(
        slurm.get("parallel_online_total") == {"gpus": 6, "cpus": 48, "memory_gib": 192}
        and slurm.get("parallel_online_plus_service_total") == {"gpus": 6, "cpus": 72, "memory_gib": 288},
        "M5 parallel resource totals drift",
    )
    _require(
        slurm.get("shared_environment_service")
        == {
            "gpus": 0,
            "cpus": 24,
            "memory_gib": 96,
            "port": 44151,
            "initial_workers": 8,
            "worker_candidates": [8, 16],
            "selected_workers": 16,
            "request_concurrency": "one in-flight HTTP request per worker process",
            "selection_reason": "both candidates had zero failures; 16 workers preserves more independent capacity for six concurrent online runs while 32 lanes avoided the doubled tail latency seen at 64",
            "renewable": True,
            "renewal_mechanism": "sbatch_successor_afterany",
            "cuda_visible_devices": "empty",
        },
        "M5 shared service resource contract drift",
    )
    server_runtime = protocol.get("server_runtime")
    _require(isinstance(server_runtime, Mapping), "M5 server runtime contract is missing")
    _require(server_runtime.get("python") == "3.12.13", "M5 server Python drift")
    _require(server_runtime.get("java") == "21.0.10", "M5 server Java drift")
    _require(
        server_runtime.get("request_concurrency")
        == {
            "mode": "process_serialized_asgi_v1",
            "reason": "the pinned upstream keeps one SQLite connection, Lucene searcher and mutable product cache per worker while FastAPI runs sync endpoints in a thread pool",
            "health_probe": "concurrent connection-closing waves must cover every worker PID",
            "maximum_http_5xx_fraction": 0.0,
        },
        "M5 server request-concurrency contract drift",
    )
    _require(
        server_runtime.get("source_fetch_policy")
        == {
            "git_attempts": 2,
            "git_attempt_timeout_seconds": 60,
            "locked_archive_attempts": 4,
        },
        "M5 server source-fetch policy drift",
    )
    _require(
        server_runtime.get("critical_packages")
        == {
            "pandas": "3.0.3",
            "pyarrow": "25.0.0",
            "fastapi": "0.139.2",
            "gunicorn": "26.0.0",
            "uvicorn": "0.51.0",
            "pyserini": "2.3.0",
            "pyjnius": "1.7.0",
            "httpx": "0.28.1",
            "rank-bm25": "0.2.2",
        },
        "M5 server package drift",
    )
    _require(
        server_runtime.get("reward_changing_optional_packages_forbidden") == ["spacy", "thefuzz"],
        "M5 server optional-package policy drift",
    )
    _require(
        server_runtime.get("allocation_consistency")
        == "every service allocation audit content_sha256 must equal the setup audit",
        "M5 server allocation-consistency drift",
    )
    training_runtime = protocol.get("training_runtime")
    _require(isinstance(training_runtime, Mapping), "M5 training runtime contract is missing")
    _require(training_runtime.get("python_major_minor") == "3.11", "M5 training Python drift")
    _require(
        training_runtime.get("critical_packages")
        == {
            "torch": "2.10.0",
            "transformers": "5.14.1",
            "vllm": "0.17.0",
            "peft": "0.19.1",
            "requests": "2.32.5",
            "urllib3": "2.6.3",
            "chardet": "5.2.0",
            "charset-normalizer": "3.4.5",
        },
        "M5 training runtime package drift",
    )
    _require(
        training_runtime.get("known_pip_check_exception")
        == "vllm 0.17.0 has requirement transformers<5,>=4.56.0, but you have transformers 5.14.1.",
        "M5 training runtime metadata-exception drift",
    )
    _require(
        training_runtime.get("exception_scope")
        == "metadata-only exception accepted for preflight because vLLM 0.17.0 natively registers Qwen3_5ForConditionalGeneration; formal release still requires real deterministic generation, behavior-logprob, sleep/wake and learner-update gates",
        "M5 training runtime exception-scope drift",
    )
    _require(
        training_runtime.get("requests_warning_policy") == "RequestsDependencyWarning is forbidden",
        "M5 requests warning policy drift",
    )
    load_upstream_lock(PROJECT_ROOT / sources["agent_r1_data"]["lock_path"])
    return protocol


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = validate_protocol(_json(resolved))
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "git_sha": repository_git_sha(),
        "payload": payload,
    }


def split_for_goal_index(goal_index: int) -> str:
    if not isinstance(goal_index, int) or isinstance(goal_index, bool) or not 0 <= goal_index < 12087:
        raise ValueError("goal index outside frozen WebShop roster")
    if goal_index < 500:
        return "test"
    if goal_index < 1000:
        return "dev"
    return "train"


def task_id_for_goal_index(goal_index: int) -> str:
    split_for_goal_index(goal_index)
    return f"webshop_goal_{goal_index:05d}"


def normalized_instruction(instruction: str) -> str:
    _require(isinstance(instruction, str) and instruction.strip(), "instruction is empty")
    return " ".join(instruction.lower().split())


def eligible_goal_indices(
    split: str,
    exclusions: Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    bounds = {"test": (0, 500), "dev": (500, 1000), "train": (1000, 12087)}
    _require(split in bounds, f"unknown M5 split: {split}")
    payload = dict(exclusions or load_split_exclusions()["payload"])
    validate_split_exclusions(payload)
    start, stop = bounds[split]
    blocked = set(payload["excluded_goal_indices"][split])
    return tuple(index for index in range(start, stop) if index not in blocked)


def deterministic_selection(*, start: int, stop: int, count: int, seed: int, namespace: str) -> tuple[int, ...]:
    _require(0 <= start < stop <= 12087, "invalid deterministic roster bounds")
    _require(0 < count <= stop - start, "invalid deterministic roster count")
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, "invalid roster seed")
    _require(isinstance(namespace, str) and namespace, "roster namespace is missing")
    ranked = sorted(
        range(start, stop),
        key=lambda index: (
            hashlib.sha256(f"{namespace}|{seed}|{index}".encode("utf-8")).digest(),
            index,
        ),
    )
    return tuple(sorted(ranked[:count]))


def deterministic_candidate_selection(
    *,
    candidates: tuple[int, ...],
    count: int,
    seed: int,
    namespace: str,
) -> tuple[int, ...]:
    _require(candidates == tuple(sorted(set(candidates))), "candidate roster is not unique and sorted")
    _require(0 < count <= len(candidates), "invalid candidate selection count")
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, "invalid roster seed")
    _require(isinstance(namespace, str) and namespace, "roster namespace is missing")
    ranked = sorted(
        candidates,
        key=lambda index: (
            hashlib.sha256(f"{namespace}|{seed}|{index}".encode("utf-8")).digest(),
            index,
        ),
    )
    return tuple(sorted(ranked[:count]))


def deterministic_candidate_order(
    *,
    candidates: tuple[int, ...],
    seed: int,
    namespace: str,
) -> tuple[int, ...]:
    _require(candidates == tuple(sorted(set(candidates))), "candidate roster is not unique and sorted")
    _require(bool(candidates), "candidate roster is empty")
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, "invalid roster seed")
    _require(isinstance(namespace, str) and namespace, "roster namespace is missing")
    return tuple(
        sorted(
            candidates,
            key=lambda index: (
                hashlib.sha256(f"{namespace}|{seed}|{index}".encode("utf-8")).digest(),
                index,
            ),
        )
    )


def sft_candidate_orders(protocol: Mapping[str, Any] | None = None) -> dict[str, tuple[int, ...]]:
    """Return full scan orders; the final roster depends on verified replay."""

    payload = dict(protocol or load_protocol()["payload"])
    sft = payload["sft"]
    seed = int(sft["selection_seed"])
    train = deterministic_candidate_order(
        candidates=eligible_goal_indices("train"), seed=seed, namespace="m5-sft-train"
    )
    dev = deterministic_candidate_order(
        candidates=eligible_goal_indices("dev"), seed=seed, namespace="m5-sft-dev"
    )
    _require(not set(train) & set(dev), "M5 SFT train/dev overlap")
    _require(all(split_for_goal_index(index) == "train" for index in train), "M5 SFT train role drift")
    _require(all(split_for_goal_index(index) == "dev" for index in dev), "M5 SFT dev role drift")
    return {"train": train, "dev": dev}


def audit_goals(goals_path: Path, protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(protocol or load_protocol()["payload"])
    path = Path(goals_path).expanduser().resolve()
    expected_sha = next(
        item["sha256"]
        for item in load_upstream_lock()["payload"]["runtime_files"]
        if item["path"] == "goals.json"
    )
    _require(path.is_file(), f"WebShop goals are missing: {path}")
    _require(sha256_file(path) == expected_sha, "WebShop goals SHA256 drift")
    goals = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(goals, list) and len(goals) == payload["dataset"]["goals"], "WebShop goal count drift")
    instructions: dict[str, list[int]] = {}
    categories: dict[str, int] = {}
    for index, goal in enumerate(goals):
        _require(isinstance(goal, Mapping), f"invalid goal row: {index}")
        _require(goal.get("goal_index") == index, f"non-contiguous goal index: {index}")
        _require(isinstance(goal.get("instruction"), str) and goal["instruction"].strip(), f"missing instruction: {index}")
        _require(isinstance(goal.get("asin"), str) and goal["asin"], f"missing ASIN: {index}")
        _require(isinstance(goal.get("query"), str) and goal["query"], f"missing query: {index}")
        _require(goal.get("reward_mode") == "webshop_full", f"reward mode drift: {index}")
        normalized = normalized_instruction(goal["instruction"])
        instructions.setdefault(normalized, []).append(index)
        category = str(goal.get("category") or "unknown")
        categories[category] = categories.get(category, 0) + 1
    later_duplicate_rows = len(goals) - len(instructions)
    duplicate_instruction_groups = sum(len(indices) > 1 for indices in instructions.values())
    raw_sets = {
        "test": {normalized_instruction(goal["instruction"]) for goal in goals[0:500]},
        "dev": {normalized_instruction(goal["instruction"]) for goal in goals[500:1000]},
        "train": {normalized_instruction(goal["instruction"]) for goal in goals[1000:12087]},
    }
    raw_intersections = {
        "test_dev": len(raw_sets["test"] & raw_sets["dev"]),
        "test_train": len(raw_sets["test"] & raw_sets["train"]),
        "dev_train": len(raw_sets["dev"] & raw_sets["train"]),
    }
    split_lock = load_split_exclusions()["payload"]
    _require(
        raw_intersections == split_lock["raw_cross_role_normalized_instruction_intersections"],
        "M5 raw cross-role instruction overlap drift",
    )
    seen: set[str] = set()
    observed_exclusions = {"test": [], "dev": [], "train": []}
    for index, goal in enumerate(goals):
        split = split_for_goal_index(index)
        key = normalized_instruction(goal["instruction"])
        if split != "test" and key in seen:
            observed_exclusions[split].append(index)
        seen.add(key)
    _require(
        observed_exclusions == split_lock["excluded_goal_indices"],
        "M5 canonical-first split exclusions drift",
    )
    eligible_sets = {
        split: {normalized_instruction(goals[index]["instruction"]) for index in eligible_goal_indices(split, split_lock)}
        for split in ("test", "dev", "train")
    }
    _require(not eligible_sets["test"] & eligible_sets["dev"], "M5 eligible test/dev instruction overlap")
    _require(not eligible_sets["test"] & eligible_sets["train"], "M5 eligible test/train instruction overlap")
    _require(not eligible_sets["dev"] & eligible_sets["train"], "M5 eligible dev/train instruction overlap")
    candidate_orders = sft_candidate_orders(payload)
    return {
        "schema_version": "m5_webshop_goal_audit_v1",
        "passed": True,
        "goals_path": str(path),
        "goals_sha256": expected_sha,
        "goal_count": len(goals),
        "raw_split_counts": {"test": 500, "dev": 500, "train": 11087},
        "eligible_split_counts": {split: len(eligible_goal_indices(split, split_lock)) for split in ("test", "dev", "train")},
        "raw_cross_role_normalized_instruction_intersections": raw_intersections,
        "eligible_cross_role_normalized_instruction_intersections": {"test_dev": 0, "test_train": 0, "dev_train": 0},
        "unique_normalized_instructions": len(instructions),
        "later_duplicate_normalized_instruction_rows": later_duplicate_rows,
        "duplicate_normalized_instruction_group_count": duplicate_instruction_groups,
        "category_counts": dict(sorted(categories.items())),
        "sft_train_candidate_count": len(candidate_orders["train"]),
        "sft_dev_candidate_count": len(candidate_orders["dev"]),
        "sft_train_candidate_order_sha256": _compact_index_sha256(list(candidate_orders["train"])),
        "sft_dev_candidate_order_sha256": _compact_index_sha256(list(candidate_orders["dev"])),
        "test_access_policy": "mechanical integrity and duplicate-isolation audit only; no model update, selection, or preflight metric",
    }
