import copy

import pytest

from miniwebwork.long_horizon_rl.sft_selection import (
    load_sft_preflight_selection,
    validate_sft_preflight_selection,
)


def test_passing_sft_preflight_selection_is_frozen_and_disposable():
    loaded = load_sft_preflight_selection()
    payload = loaded["payload"]
    assert payload["result"] == "PASS"
    assert payload["formal_training"] is False
    assert payload["disposable_adapter"] is True
    assert payload["slurm"]["job_id"] == 1264
    assert payload["selection"]["selected_microbatch"] == 8
    assert payload["selection"]["gradient_accumulation_steps"] == 2
    assert payload["training_smoke"]["optimizer_updates"] == 20
    assert payload["telemetry"]["external_vram_headroom_fraction"] >= 0.15


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("formal_training",), True, "mislabeled"),
        (("disposable_adapter",), False, "disposable"),
        (("slurm", "job_id"), 1263, "JobID"),
        (("selection", "selected_microbatch"), 4, "microbatch"),
        (("selection", "selected_reserved_vram_headroom_fraction"), 0.14, "headroom"),
        (("training_smoke", "optimizer_updates"), 19, "update"),
        (("adapter_audit", "all_finite"), False, "non-finite"),
    ],
)
def test_sft_preflight_selection_fails_closed_on_evidence_drift(path, value, error):
    payload = copy.deepcopy(load_sft_preflight_selection()["payload"])
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError, match=error):
        validate_sft_preflight_selection(payload)
