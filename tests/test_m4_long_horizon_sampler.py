from miniwebwork.long_horizon_rl.sampler import (
    DeterministicSignalSampler,
    TaskDescriptor,
)


def _tasks():
    families = (
        ("exact_product", "basic"),
        ("cheapest_feasible", "medium"),
        ("no_feasible_product", "medium"),
        ("highest_reliability_supplier", "long"),
    )
    return [
        TaskDescriptor(f"TASK-{family}-{index:02d}", family, horizon)
        for family, horizon in families
        for index in range(10)
    ]


def test_sampler_is_seeded_reproducible_and_family_balanced():
    first = DeterministicSignalSampler(_tasks(), study_seed=20260801)
    second = DeterministicSignalSampler(_tasks(), study_seed=20260801)
    selection = first.select(iteration_index=0, limit=8)
    assert selection == second.select(iteration_index=0, limit=8)
    assert selection != DeterministicSignalSampler(
        _tasks(), study_seed=20260802
    ).select(iteration_index=0, limit=8)
    counts = {}
    for task in selection:
        counts[task.task_family] = counts.get(task.task_family, 0) + 1
    assert set(counts.values()) == {2}


def test_sampler_completes_cold_coverage_before_revisiting_family_tasks():
    sampler = DeterministicSignalSampler(_tasks(), study_seed=20260801)
    first = sampler.select(iteration_index=0, limit=8)
    for task in first:
        sampler.record_committed_group(task.task_id, [1.0, 0.0, 0.0, 1.0])
    second = sampler.select(iteration_index=1, limit=8)
    assert not ({task.task_id for task in first} & {task.task_id for task in second})


def test_sampler_prefers_uncertain_signal_after_full_cold_coverage():
    sampler = DeterministicSignalSampler(_tasks(), study_seed=20260801)
    for task in _tasks():
        # Mark every task seen and mostly saturated-success.
        sampler.record_committed_group(task.task_id, [1.0, 1.0, 1.0, 1.0])
        sampler.record_committed_group(task.task_id, [1.0, 1.0, 1.0, 1.0])
    uncertain_ids = set()
    for family in sorted({task.task_family for task in _tasks()}):
        task = next(task for task in _tasks() if task.task_family == family)
        # Additional mixed groups move these tasks close to p=0.5.
        for _ in range(4):
            sampler.record_committed_group(task.task_id, [0.0, 0.0, 0.0, 0.0])
        uncertain_ids.add(task.task_id)
    selection = sampler.select(iteration_index=9, limit=4)
    assert {task.task_id for task in selection} == uncertain_ids


def test_sampler_accounts_infrastructure_attempts_without_treating_them_as_rewards():
    sampler = DeterministicSignalSampler(_tasks(), study_seed=20260801)
    task_id = _tasks()[0].task_id
    sampler.record_infra_invalid_attempt(task_id)
    audit = sampler.audit_dict()["signals"][task_id]
    assert audit["infra_invalid_attempts"] == 1
    assert audit["valid_trajectories"] == 0
    assert audit["posterior_success_probability"] == 0.5
