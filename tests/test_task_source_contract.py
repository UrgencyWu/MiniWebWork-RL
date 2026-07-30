import json

import pytest

from miniwebwork.tasks import (
    TaskDataError,
    get_public_task,
    load_oracle_tasks,
    load_public_tasks,
)


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_explicit_task_source_is_exclusive(tmp_path, monkeypatch):
    env_dir = tmp_path / "env_source"
    explicit_dir = tmp_path / "explicit_source"
    env_dir.mkdir()
    explicit_dir.mkdir()

    _write_jsonl(env_dir / "valid_public.jsonl", [{"task_id": "ENV-1", "instruction": "env"}])
    _write_jsonl(env_dir / "valid_oracle.jsonl", [{"task_id": "ENV-1", "objective": "x"}])
    _write_jsonl(
        explicit_dir / "valid_public.jsonl",
        [{"task_id": "EXPLICIT-1", "instruction": "explicit"}],
    )
    _write_jsonl(
        explicit_dir / "valid_oracle.jsonl",
        [{"task_id": "EXPLICIT-1", "objective": "x"}],
    )

    monkeypatch.setenv("MINIWEBWORK_TASK_DIR", str(env_dir))

    assert [task["task_id"] for task in load_public_tasks(explicit_dir)] == ["EXPLICIT-1"]
    assert [task["task_id"] for task in load_oracle_tasks(explicit_dir)] == ["EXPLICIT-1"]
    assert get_public_task("ENV-1", task_dir=explicit_dir) is None


def test_environment_task_source_is_not_merged_with_default(tmp_path, monkeypatch):
    _write_jsonl(tmp_path / "train_public.jsonl", [{"task_id": "PATCH-1"}])
    _write_jsonl(tmp_path / "train_oracle.jsonl", [{"task_id": "PATCH-1"}])
    monkeypatch.setenv("MINIWEBWORK_TASK_DIR", str(tmp_path))

    assert [task["task_id"] for task in load_public_tasks()] == ["PATCH-1"]
    assert [task["task_id"] for task in load_oracle_tasks()] == ["PATCH-1"]


def test_duplicate_task_ids_fail_fast(tmp_path):
    _write_jsonl(tmp_path / "train_public.jsonl", [{"task_id": "DUP-1"}])
    _write_jsonl(tmp_path / "valid_public.jsonl", [{"task_id": "DUP-1"}])
    _write_jsonl(tmp_path / "train_oracle.jsonl", [{"task_id": "DUP-1"}])

    with pytest.raises(TaskDataError, match="Duplicate public task_id"):
        load_public_tasks(tmp_path)


def test_missing_task_files_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .*public"):
        load_public_tasks(tmp_path)
