from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _dropout_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m6_phase2_dropout_probe.py"
    spec = importlib.util.spec_from_file_location("m6_phase2_dropout_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _readiness_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m6_phase2_buy_readiness.py"
    spec = importlib.util.spec_from_file_location("m6_phase2_buy_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _batch_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m6_phase2_batch_probe.py"
    spec = importlib.util.spec_from_file_location("m6_phase2_batch_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reward_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m6_phase2_reward_probe.py"
    spec = importlib.util.spec_from_file_location("m6_phase2_reward_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _collector_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m6_collect_policy_success.py"
    spec = importlib.util.spec_from_file_location("m6_phase2_collect", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase2_coefficient_of_variation():
    module = _dropout_module()
    assert module._coefficient_of_variation([2.0, 2.0, 2.0]) == 0.0
    assert module._coefficient_of_variation([1.0, 3.0]) == pytest.approx(0.5)


def test_phase2_gradient_cosine_is_bounded_and_precise():
    torch = pytest.importorskip("torch")
    module = _dropout_module()
    left = torch.ones(2_000_003, dtype=torch.float32)
    right = left.clone()
    assert module._gradient_cosine(left, right, torch, chunk_size=100_000) == 1.0
    assert module._gradient_cosine(left, -right, torch, chunk_size=100_000) == -1.0


def test_phase2_dropout_probe_has_no_optimizer_step():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "m6_phase2_dropout_probe.py"
    ).read_text(encoding="utf-8")
    assert "optimizer.step(" not in source
    assert '"optimizer_steps": 0' in source


def test_phase2_dropout_probe_forces_only_dropout_modules_to_eval():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "m6_phase2_dropout_probe.py"
    ).read_text(encoding="utf-8")
    assert "model.train()" in source
    assert "isinstance(module, torch.nn.Dropout)" in source
    assert "module.eval()" in source


def test_phase2_job_is_bounded_and_single_gpu():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_m6_phase2_dropout_probe_job.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --time=02:00:00" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=24G" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "test ! -e \"$output\"" in source


def _evidence(*, page_type: str, item: float, option: float = 0.0, option_count: int = 0):
    module = _readiness_module()
    value = {
        "policy_visible_input_only": True,
        "gradient_attached": False,
        "boundary": "nonterminal_public_state",
        "page_type": page_type,
        "item_attribute_match_fraction": item,
        "selected_option_fraction": option,
        "selected_option_count": option_count,
    }
    value["content_sha256"] = module._self_hash(value)
    return value


def test_phase2_readiness_ignores_search_results():
    module = _readiness_module()
    assert module._readiness_from_evidence(_evidence(page_type="search_results", item=1.0)) == 0.0


def test_phase2_readiness_reserves_option_weight_only_when_required():
    module = _readiness_module()
    assert module._readiness_from_evidence(_evidence(page_type="item", item=0.75)) == pytest.approx(0.75)
    assert module._readiness_from_evidence(
        _evidence(page_type="item", item=0.75, option=0.5, option_count=2)
    ) == pytest.approx(0.7)


def test_phase2_readiness_v2_uses_equal_semantic_blocks():
    module = _readiness_module()
    evidence = _evidence(page_type="item", item=0.75, option=0.5, option_count=2)
    assert module._readiness_from_evidence(
        evidence,
        formula_version=module.EQUAL_BLOCK_FORMULA,
    ) == pytest.approx(0.625)
    assert module._readiness_from_evidence(
        _evidence(page_type="item", item=0.75),
        formula_version=module.EQUAL_BLOCK_FORMULA,
    ) == pytest.approx(0.75)


def test_phase2_readiness_v2_job_is_cpu_only_and_bound_to_full_horizon():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_m6_phase2_buy_readiness_v2_job.sh").read_text(encoding="utf-8")
    submit = (root / "scripts" / "submit_m6_phase2_buy_readiness_v2.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu" not in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=8G" in source
    assert "--identity full_18_15_prefix_replay_r2" in source
    assert "--formula-version equal_item_option_blocks_v2" in source
    assert '"decision":"USE_FULL_HORIZON"' in source
    assert "git status --porcelain --untracked-files=no" in submit


def test_phase2_readiness_auc_handles_ties():
    module = _readiness_module()
    assert module._auc([0, 1], [0.5, 0.5]) == 0.5


def test_phase2_reward_probe_is_strict_dominant():
    module = _reward_module()
    evidence = _evidence(page_type="item", item=0.8, option=0.0, option_count=1)
    partial = {
        "success": False,
        "task_score": 0.5,
        "termination_reason": "purchase",
        "turns": [
            {"verifier_progress_evidence": evidence},
            {"verifier_progress_evidence": evidence},
        ],
    }
    strict = dict(partial, success=True, task_score=1.0)
    partial_reward = module.trajectory_rewards(partial)
    strict_reward = module.trajectory_rewards(strict)
    assert -0.1 <= partial_reward["strict_dominant_reward"] <= 0.0
    assert strict_reward["strict_dominant_reward"] == 1.0
    assert strict_reward["strict_dominant_reward"] > partial_reward["strict_dominant_reward"]


def test_phase2_reward_probe_has_no_optimizer_step_and_is_bounded():
    root = Path(__file__).resolve().parents[1]
    probe = (root / "scripts" / "m6_phase2_reward_probe.py").read_text(encoding="utf-8")
    job = (root / "scripts" / "run_m6_phase2_reward_probe_job.sh").read_text(encoding="utf-8")
    submit = (root / "scripts" / "submit_m6_phase2_reward_probe.sh").read_text(encoding="utf-8")
    assert "optimizer.step(" not in probe
    assert '"optimizer_steps": 0' in probe
    assert "official_dense_task_score_used_as_reward" in probe
    assert "#SBATCH --gres=gpu:1" in job
    assert "#SBATCH --cpus-per-task=4" in job
    assert "#SBATCH --mem=24G" in job
    assert "full_18_15_prefix_replay_r2/groups" in job
    assert '"process_reward_calibration_passed":true' in job
    assert "git status --porcelain --untracked-files=no" in submit


def test_phase2_p0_submit_has_no_false_dependency():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "submit_m6_phase2_p0.sh"
    ).read_text(encoding="utf-8")
    assert source.count("sbatch --parsable") == 2
    assert "--dependency" not in source
    assert "git status --porcelain --untracked-files=no" in source


def test_phase2_batch_gradient_cosine_is_bounded():
    torch = pytest.importorskip("torch")
    module = _batch_module()
    left = torch.ones(1_000_003, dtype=torch.float32)
    assert module._gradient_cosine(left, left.clone(), torch, chunk_size=100_000) == 1.0


def test_phase2_batch_job_is_bounded_and_single_gpu():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_m6_phase2_batch_probe_job.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --time=02:00:00" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=24G" in source
    assert "#SBATCH --gres=gpu:1" in source


def test_phase2_horizon_contract_allows_only_two_frozen_arms():
    module = _collector_module()
    from argparse import Namespace

    args = Namespace(
        role="train",
        task_roster=Path("roster.json"),
        adapter=Path("adapter"),
        max_model_turns=6,
        max_environment_steps=6,
        maximum_tasks=None,
        task_offset=0,
        maximum_action_tokens=None,
        replay_prefix_root=None,
    )
    module.validate_phase2_horizon_contract(args)
    args.max_model_turns = 18
    args.max_environment_steps = 15
    args.replay_prefix_root = Path("short")
    module.validate_phase2_horizon_contract(args)
    args.max_model_turns = 6
    args.max_environment_steps = 6
    with pytest.raises(ValueError, match="only valid for the full-horizon arm"):
        module.validate_phase2_horizon_contract(args)
    args.replay_prefix_root = None
    args.max_model_turns = 12
    with pytest.raises(ValueError, match="horizon arm drift"):
        module.validate_phase2_horizon_contract(args)


def test_phase2_prefix_replay_preserves_audited_generation_evidence():
    module = _collector_module()

    class LiveBackend:
        def generate(self, messages):
            return ("live", messages)

    lineage = {
        "adapter_sha256": "a" * 64,
        "rollout_adapter_sha256": "b" * 64,
        "adapter_semantic_sha256": "c" * 64,
    }
    source = {
        "prompt_token_ids": [1, 2],
        "generated_token_ids": [3, 4],
        "behavior_logprobs": [-0.1, -0.2],
        "sampling_logprobs": [-0.1, -0.2],
        "raw_output": '{"command":"search[test]"}',
        "request_id": "source.request",
        "sampling_seed": 7,
    }
    backend = module.Phase2PrefixReplayBackend(
        live_backend=LiveBackend(), source_turns=[source], lineage=lineage
    )
    replay = backend.generate([{"role": "user", "content": "prompt"}])
    assert replay.prompt_token_ids == [1, 2]
    assert replay.generated_token_ids == [3, 4]
    assert replay.logprobs == [-0.1, -0.2]
    assert replay.sampling_seed == 7
    assert replay.generation_backend == "phase2_prefix_replay_v1"
    assert backend.consumed_prefix_turns == 1
    assert backend.generate([{"role": "user", "content": "next"}]) == (
        "live",
        [{"role": "user", "content": "next"}],
    )


def test_phase2_horizon_jobs_are_bounded_and_paired():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_m6_phase2_horizon_job.sh").read_text(encoding="utf-8")
    audit = (root / "scripts" / "run_m6_phase2_horizon_analysis_job.sh").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1%2" in source
    assert "--seed 20260821" in source
    assert "--mode phase2_horizon_evaluation" in source
    assert "model_turns=6; environment_steps=6" in source
    assert "model_turns=18; environment_steps=15" in source
    assert "#SBATCH --gres=gpu" not in audit


def test_phase2_horizon_roster_uses_only_train_role():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "m6_phase2_build_horizon_roster.py"
    ).read_text(encoding="utf-8")
    assert 'split["roles"]["train"]["task_ids"]' in source
    assert 'if role != "train"' in source
    assert "eligible_ids = [task_id for task_id in train_ids if task_id not in exposed_ids]" in source
    assert 'not (set(selected) & exposed_ids)' in source
    assert '"excluded_overlap_task_count": len(excluded_overlap_ids)' in source
    assert '"task_count": len(selected)' in source


def test_phase2_horizon_submit_builds_roster_before_jobs():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "submit_m6_phase2_horizon.sh"
    ).read_text(encoding="utf-8")
    build = source.index("m6_phase2_build_horizon_roster.py")
    submit = source.index("horizon_job=")
    assert build < submit
    assert 'dependency="afterok:$horizon_job"' in source
    assert "M6_SERVICE_BASE_URL" in source


def test_phase2_horizon_replay_is_single_gpu_and_audited_afterok():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_m6_phase2_horizon_replay_job.sh").read_text(encoding="utf-8")
    audit = (root / "scripts" / "run_m6_phase2_horizon_replay_analysis_job.sh").read_text(encoding="utf-8")
    submit = (root / "scripts" / "submit_m6_phase2_horizon_replay.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in source
    assert "--replay-prefix-root" in source
    assert "initial_turn_index=len(source_turns)" in (
        root / "scripts" / "m6_collect_policy_success.py"
    ).read_text(encoding="utf-8")
    assert "--max-model-turns 18 --max-environment-steps 15" in source
    assert "#SBATCH --gres=gpu" not in audit
    assert 'dependency="afterok:$replay_job"' in submit
    assert "full_18_15_prefix_replay_r2" in source


def test_phase2_horizon_analysis_requires_replay_source_binding():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "m6_phase2_horizon_analysis.py"
    ).read_text(encoding="utf-8")
    assert 'full_report.get("replay_prefix_source")' in source
    assert 'replay_source.get("collection_report_content_sha256")' in source
    assert 'full_invocation.get("replay_prefix_source") == replay_source' in source
