from pathlib import Path

import pytest

from miniwebwork.sft.m4_dataset import build_m4_oracle_sft_dataset


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "data" / "tasks" / "m4_rlvr_v1"
SEED_DIR = ROOT / "data" / "seed_m4_rlvr_v1"


def test_m4_sft_builder_rejects_non_training_splits_before_browser_work(tmp_path: Path):
    with pytest.raises(PermissionError, match="cannot be used for 'offline_training'"):
        build_m4_oracle_sft_dataset(
            tmp_path / "out",
            task_dir=TASK_ROOT / "test",
            seed_dir=SEED_DIR,
            task_limit=1,
        )


def test_m4_sft_builder_rejects_invalid_limits_before_browser_work(tmp_path: Path):
    with pytest.raises(ValueError, match="task_limit must be positive"):
        build_m4_oracle_sft_dataset(
            tmp_path / "out",
            task_dir=TASK_ROOT / "train",
            seed_dir=SEED_DIR,
            task_limit=0,
        )
