#!/usr/bin/env python3
"""Generate two public queries per nav task with Qwen3.5-35B and verify a Specialist corpus."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.long_horizon_rl.model_manifest import validate_base_model_manifest  # noqa: E402
from miniwebwork.long_horizon_rl.vllm_backend import RawAsyncVLLMGenerationEngine, RawVLLMBackendConfig  # noqa: E402
from miniwebwork.m6_phase10c_data import validate_phase10c_split  # noqa: E402
from miniwebwork.m6_phase10c_specialist_data import build_specialist_smoke_corpus  # noqa: E402
from miniwebwork.webshop_rl.actions import MAX_SEARCH_QUERY_CHARACTERS  # noqa: E402
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment  # noqa: E402

QUERY_FORMULA = "qwen3.5-35b-a3b_public_instruction_k2_v1"
ASIN_RE = re.compile(r"\bB[0-9A-Z]{9}\b", re.IGNORECASE)
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_query_completion(raw_text: str, *, instruction: str) -> str:
    text = str(raw_text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    _require(start >= 0 and end >= start, "Phase10-C query generator returned no JSON object")
    payload = json.loads(text[start : end + 1])
    _require(isinstance(payload, Mapping) and set(payload) == {"query"},
             "Phase10-C query generator JSON keys drift")
    query = " ".join(str(payload["query"] or "").strip().split())
    _require(query and len(query) <= MAX_SEARCH_QUERY_CHARACTERS, "Phase10-C generated query length drift")
    _require("[" not in query and "]" not in query, "Phase10-C generated query contains delimiters")
    _require(
        ASIN_RE.search(query) is None or ASIN_RE.search(instruction) is not None,
        "Phase10-C generated query introduced an unseen ASIN",
    )
    return query


async def generate_queries(
    *,
    base_model: Path,
    base_manifest: Path,
    goals_by_task: Mapping[str, Mapping[str, Any]],
    task_ids: list[str],
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    manifest = validate_base_model_manifest(
        base_manifest,
        expected_base_model=base_model,
        # Phase10-B already performed the expensive full-file verification for
        # this exact manifest. This smoke rebinds the immutable manifest and
        # avoids rehashing the full 35B checkpoint.
        verify_files=False,
    )
    engine = await RawAsyncVLLMGenerationEngine.create(RawVLLMBackendConfig(
        base_model=str(base_model),
        base_model_manifest_sha256=manifest["sha256"],
        base_model_functional_sha256=manifest["payload"]["functional_file_set_sha256"],
        seed=seed,
        tensor_parallel_size=2,
    ))
    records: list[dict[str, Any]] = []

    async def one(task_id: str, sample_index: int) -> tuple[str, int, Any]:
        instruction = str(goals_by_task[task_id]["instruction"])
        messages = [
            {"role": "system", "content": (
                "Create one concise WebShop search query from the public shopping instruction. "
                "Do not output an ASIN, explanation, markdown, or brackets. Return exactly JSON: {\"query\":\"...\"}."
            )},
            {"role": "user", "content": instruction},
        ]
        result = await engine.generate_messages(
            messages,
            request_id=f"p10c-query-{task_id}-{sample_index}",
            sampling_seed=seed + int(task_id.rsplit("_", 1)[1]) * 2 + sample_index,
        )
        return task_id, sample_index, result

    try:
        results = await asyncio.gather(*(one(task_id, sample) for task_id in task_ids for sample in range(2)))
    finally:
        engine.shutdown()
    queries: dict[str, list[str]] = {task_id: [] for task_id in task_ids}
    invalid_counts: dict[str, int] = {}
    for task_id, sample_index, result in results:
        instruction = str(goals_by_task[task_id]["instruction"])
        error = ""
        query = ""
        if result.error:
            error = "backend_error"
        else:
            try:
                query = parse_query_completion(result.raw_text, instruction=instruction)
            except (ValueError, json.JSONDecodeError) as exc:
                error = type(exc).__name__
        if query and query not in queries[task_id]:
            queries[task_id].append(query)
        if error:
            invalid_counts[error] = invalid_counts.get(error, 0) + 1
        records.append({
            "task_id": task_id,
            "sample_index": sample_index,
            "request_id": result.request_id,
            "sampling_seed": result.sampling_seed,
            "generated_query": query,
            "raw_text_sha256": sha256_json({"raw_text": result.raw_text}),
            "error_class": error,
        })
    tasks_without_candidates = [task_id for task_id in task_ids if not queries[task_id]]
    report = {
        "schema_version": "m6_phase10c_qwen35_public_query_generation_v1",
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "model": str(base_model),
        "base_model_manifest_sha256": manifest["sha256"],
        "query_formula": QUERY_FORMULA,
        "task_count": len(task_ids),
        "attempt_count": len(records),
        "valid_unique_query_count": sum(len(value) for value in queries.values()),
        "task_count_with_valid_query": len(task_ids) - len(tasks_without_candidates),
        "tasks_without_valid_query": tasks_without_candidates,
        "invalid_counts": dict(sorted(invalid_counts.items())),
        "queries_by_task": queries,
        "records": records,
    }
    report["content_sha256"] = sha256_json(report)
    return queries, report


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{os.urandom(4).hex()}")
    content = b"".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--phase10c-split", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--base-model-manifest", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260861)
    args = parser.parse_args()

    current_git = _git_sha()
    _require(GIT_SHA_RE.fullmatch(current_git) is not None and args.producer_git_sha == current_git,
             "Phase10-C query smoke Git drift")
    output = args.output_dir.expanduser().resolve()
    _require(not output.exists(), "Phase10-C Qwen35 query smoke output already exists")
    split = validate_phase10c_split(json.loads(args.phase10c_split.expanduser().resolve().read_text(encoding="utf-8")))
    goals = json.loads(args.goals.expanduser().resolve().read_text(encoding="utf-8"))
    task_ids = list(split["roles"]["teacher_nav_train"]["task_ids"][:16])
    goals_by_task = {f"webshop_goal_{int(goal['goal_index']):05d}": goal for goal in goals}
    queries, query_report = asyncio.run(generate_queries(
        base_model=args.base_model.expanduser().resolve(),
        base_manifest=args.base_model_manifest.expanduser().resolve(),
        goals_by_task=goals_by_task,
        task_ids=task_ids,
        seed=args.seed,
    ))
    corpus = build_specialist_smoke_corpus(
        specialist="S_nav_sft",
        goals=goals,
        task_ids=task_ids,
        environment_factory=lambda: WebShopHTTPEnvironment(base_url=args.base_url, split="train", timeout_seconds=120),
        phase10c_split_content_sha256=split["content_sha256"],
        producer_git_sha=current_git,
        query_candidates_by_task=queries,
        query_formula=QUERY_FORMULA,
    )
    output.mkdir(parents=True, exist_ok=False)
    rows = list(corpus.pop("rows"))
    corpus["row_content_sha256"] = [row["content_sha256"] for row in rows]
    corpus["query_generation_content_sha256"] = query_report["content_sha256"]
    corpus["content_sha256"] = sha256_json({key: value for key, value in corpus.items() if key != "content_sha256"})
    atomic_write_json(output / "query_generation.json", query_report)
    _atomic_jsonl(output / "train.jsonl", rows)
    atomic_write_json(output / "manifest.json", corpus)
    print(json.dumps({
        "passed": corpus["passed"],
        "decision": corpus["decision"],
        "verified_tasks": corpus["verified_task_count"],
        "label_rows": corpus["label_row_count"],
        "query_attempts": query_report["attempt_count"],
        "valid_unique_queries": query_report["valid_unique_query_count"],
        "rejections": corpus["rejection_counts"],
        "output_dir": str(output),
        "manifest_sha": corpus["content_sha256"],
    }, indent=2, sort_keys=True))
    if corpus["passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
