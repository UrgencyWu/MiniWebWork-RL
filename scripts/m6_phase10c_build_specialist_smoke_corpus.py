#!/usr/bin/env python3
"""Build one 16-task Phase10-C Specialist corpus through live strict replay."""

from __future__ import annotations

import argparse
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
from miniwebwork.m6_phase10c_data import validate_phase10c_split  # noqa: E402
from miniwebwork.m6_phase10c_specialist_data import build_specialist_smoke_corpus  # noqa: E402
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment  # noqa: E402

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ROLE_BY_SPECIALIST = {
    "S_nav_sft": "teacher_nav_train",
    "S_match_sft": "teacher_match_train",
    "S_finish_sft": "teacher_finish_train",
}


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


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{os.urandom(4).hex()}")
    content = b"".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specialist", choices=sorted(ROLE_BY_SPECIALIST), required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--phase10c-split", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    current_git = _git_sha()
    _require(GIT_SHA_RE.fullmatch(current_git) is not None, "Phase10-C corpus Git SHA drift")
    _require(args.producer_git_sha == current_git, "Phase10-C corpus producer Git drift")
    output = args.output_dir.expanduser().resolve()
    _require(not output.exists(), "Phase10-C Specialist smoke output already exists")
    split = validate_phase10c_split(json.loads(args.phase10c_split.expanduser().resolve().read_text(encoding="utf-8")))
    goals = json.loads(args.goals.expanduser().resolve().read_text(encoding="utf-8"))
    _require(isinstance(goals, list) and len(goals) == 12087, "Phase10-C corpus goals drift")
    role = ROLE_BY_SPECIALIST[args.specialist]
    task_ids = list(split["roles"][role]["task_ids"][:16])
    corpus = build_specialist_smoke_corpus(
        specialist=args.specialist,
        goals=goals,
        task_ids=task_ids,
        environment_factory=lambda: WebShopHTTPEnvironment(base_url=args.base_url, split="train", timeout_seconds=120),
        phase10c_split_content_sha256=split["content_sha256"],
        producer_git_sha=current_git,
    )
    output.mkdir(parents=True, exist_ok=False)
    rows = list(corpus.pop("rows"))
    corpus["row_content_sha256"] = [row["content_sha256"] for row in rows]
    corpus["content_sha256"] = sha256_json({key: value for key, value in corpus.items() if key != "content_sha256"})
    _atomic_jsonl(output / "train.jsonl", rows)
    atomic_write_json(output / "manifest.json", corpus)
    print(json.dumps({
        "specialist": args.specialist,
        "requested_tasks": corpus["requested_task_count"],
        "verified_tasks": corpus["verified_task_count"],
        "label_rows": corpus["label_row_count"],
        "action_families": corpus["label_action_family_counts"],
        "rejections": corpus["rejection_counts"],
        "output_dir": str(output),
        "manifest_sha": corpus["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
