import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "src" / "miniwebwork" / "sft" / "train_m2_2.py"
    spec = importlib.util.spec_from_file_location("train_m2_2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completion_label_statistics_exposes_zero_label_prompt_truncation_risk():
    module = _load_module()
    dataset = [
        {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 9, 10]},
        {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, -100, -100]},
        {"input_ids": [1, 2], "labels": [-100, 7]},
    ]
    statistics = module.completion_label_token_statistics(dataset, max_length=4)
    assert statistics["sample_count"] == 3
    assert statistics["effective_supervision_sample_count"] == 2
    assert statistics["zero_completion_label_sample_count"] == 1
    assert statistics["zero_completion_label_at_max_length_sample_count"] == 1
    assert statistics["completion_tokens_per_epoch"] == 3
    assert statistics["min_nonzero_completion_label_tokens"] == 1
    assert statistics["max_nonzero_completion_label_tokens"] == 2


def test_completion_label_budget_packer_is_seeded_full_example_and_never_exceeds_target():
    module = _load_module()
    dataset = [
        {"input_ids": [1, 2], "labels": [-100, 1]},
        {"input_ids": [1, 2, 3], "labels": [-100, 1, 2]},
        {"input_ids": [1, 2, 3, 4], "labels": [-100, 1, 2, 3]},
    ]
    first, first_audit = module.pack_dataset_to_completion_token_budget(dataset, 20, 17)
    second, second_audit = module.pack_dataset_to_completion_token_budget(dataset, 20, 17)
    assert first == second
    assert first_audit == second_audit
    assert first_audit["realized_supervised_completion_tokens"] <= 20
    assert first_audit["shortfall_supervised_completion_tokens"] < 1
    assert first_audit["complete_examples_only"] is True
    assert module.completion_label_token_count(first) == first_audit["realized_supervised_completion_tokens"]
