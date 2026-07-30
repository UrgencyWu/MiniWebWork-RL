"""Canonical Qwen Base browser-Agent evaluation CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .agent_env.environment import ProcurementBrowserEnv
from .model_agent.agent_loop import run_model_episode, save_model_trajectory
from .model_agent.failure_analysis import classify_failures
from .model_agent.metrics import compute_metrics
from .model_agent.model_backend import ModelConfig, QwenTransformersBackend
from .model_agent.output_parser import parse
from .model_agent.qwen_agent import QwenBrowserAgent
from .tasks import load_public_tasks
from .model_agent import prompt_builder as pb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "m2_0"
CANONICAL_PROMPT = "browser_agent_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical Qwen Base Agent evaluation")
    parser.add_argument("--model-path", default="/data/share/model/Qwen3.5-4B")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--task-dir", type=Path, default=None)
    parser.add_argument("--max-model-turns", type=int, default=20)
    parser.add_argument("--max-env-steps", type=int, default=15)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prompt-version",
        default=CANONICAL_PROMPT,
        choices=[CANONICAL_PROMPT],
        help="The project has one active prompt contract; v1 is historical only.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.max_model_turns <= 0 or args.max_env_steps <= 0:
        raise ValueError("turn limits must be positive")
    if args.history_window < 0:
        raise ValueError("history-window must be non-negative")
    if args.limit < 0:
        raise ValueError("limit must be non-negative")

    task_dir = args.task_dir.expanduser().resolve() if args.task_dir else None
    public_tasks = load_public_tasks(task_dir)
    task_by_id = {task["task_id"]: task for task in public_tasks}
    if args.tasks == "all":
        task_ids = list(task_by_id)
    else:
        task_ids = [task_id.strip() for task_id in args.tasks.split(",") if task_id.strip()]
        unknown = [task_id for task_id in task_ids if task_id not in task_by_id]
        if unknown:
            raise ValueError(f"Unknown tasks in selected source: {unknown}")
    if args.limit:
        task_ids = task_ids[: args.limit]
    if not task_ids:
        raise ValueError("No tasks selected")

    run_id = f"canonical_base_{uuid.uuid4().hex[:8]}"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pb.PROMPT_VERSION = args.prompt_version
    pb.HISTORY_WINDOW = args.history_window
    prompt_path = pb.PROMPTS_DIR / f"{args.prompt_version}.txt"
    prompt_info = {
        "prompt_version": args.prompt_version,
        "prompt_sha256": _sha256(prompt_path),
        "prompt_builder_sha256": _sha256(Path(pb.__file__).resolve()),
        "history_window": args.history_window,
    }

    print(f"Run ID: {run_id}")
    print(f"Tasks: {len(task_ids)}")
    print(f"Task source: {task_dir or 'default'}")
    print(f"Model: {args.model_path}")
    print(f"Prompt: {args.prompt_version}")
    print(f"Output: {output_dir}")
    if args.dry_run:
        return 0

    backend = QwenTransformersBackend(
        ModelConfig(
            model_path=args.model_path,
            max_new_tokens=128,
            do_sample=False,
            enable_thinking=False,
            collect_policy_logprobs=False,
        )
    )
    task_results: list[dict] = []

    try:
        backend.load()
        model_info = backend.get_model_info()
        prompt_info["chat_template_sha256"] = backend.get_chat_template_hash()
        agent = QwenBrowserAgent(backend, pb, parse)

        with ProcurementBrowserEnv(
            max_steps=args.max_env_steps,
            run_id=run_id,
            task_dir=task_dir,
        ) as environment:
            for index, task_id in enumerate(task_ids, start=1):
                task_type = task_by_id[task_id].get("task_type", "unknown")
                print(
                    f"[{index}/{len(task_ids)}] {task_id} ({task_type})...",
                    end=" ",
                    flush=True,
                )
                started = time.time()
                result = run_model_episode(
                    task_id,
                    environment,
                    agent,
                    args.max_model_turns,
                    args.max_env_steps,
                )
                result.update(
                    run_id=run_id,
                    task_type=task_type,
                    instruction=task_by_id[task_id].get("instruction", ""),
                )
                task_results.append(result)

                status = "PASS" if result["success"] else (
                    "INFRA" if not result.get("rollout_valid", True) else "FAIL"
                )
                print(
                    f"{status} mt={result['model_turns']} "
                    f"es={result['environment_steps']} "
                    f"({time.time() - started:.1f}s) {result['termination_reason']}"
                )

                _write_json(output_dir / "per_task" / f"{task_id}.json", result)
                save_model_trajectory(
                    result,
                    output_dir / "trajectories",
                    model_info,
                    prompt_info,
                )
                raw_path = output_dir / "raw_outputs" / f"{task_id}.jsonl"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(
                    "".join(
                        json.dumps(
                            {
                                "turn": turn.get("model_turn_index", 0),
                                "raw_output": turn.get("raw_output", ""),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                        for turn in result.get("turns", [])
                    ),
                    encoding="utf-8",
                )

        metrics = compute_metrics(task_results)
        failure_analysis = classify_failures(task_results)
        manifest = {
            "schema_version": "2.0",
            "run_id": run_id,
            "git_sha": _git_sha(),
            "model_info": model_info,
            "prompt_info": prompt_info,
            "task_source": str(task_dir) if task_dir else "default",
            "task_ids": task_ids,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _write_json(output_dir / "canonical_base_metrics.json", metrics)
        _write_json(output_dir / "canonical_base_failure_analysis.json", failure_analysis)
        _write_json(output_dir / "canonical_base_manifest.json", manifest)

        print("\n=== Results ===")
        print(
            f"Success: {metrics['successful_tasks']}/{metrics['valid_tasks']} "
            f"({metrics['success_rate']:.1%}); "
            f"infrastructure={metrics['infrastructure_errors']}"
        )
        print(f"Strict JSON: {metrics['strict_json_success_rate']:.1%}")
        print(f"Schema Valid: {metrics['action_schema_valid_rate']:.1%}")
        print(f"Environment Action Success: {metrics['environment_action_success_rate']:.1%}")
        return 0 if metrics["infrastructure_errors"] == 0 else 2
    finally:
        backend.unload()


if __name__ == "__main__":
    sys.exit(main())
