"""Focused long-horizon Agent RL runtime and learner contracts.

This package is intentionally disjoint from the historical M4 algorithm-zoo
pipeline.  Its public modules are shared by the formal GRPO baseline and the
step-aware method so that collection, cost accounting, recovery, and policy
replay cannot silently diverge between methods.
"""

from .contracts import (
    COLLECTION_SCHEMA,
    GROUP_SCHEMA,
    RUN_IDENTITY_SCHEMA,
    TURN_SCHEMA,
    RunIdentity,
    canonical_public_state,
    public_anchor_signature,
    validate_committed_group,
    validate_turn_evidence,
)
from .credit import CREDIT_FORMULA_VERSION, assign_group_credit
from .journal import AppendOnlyAttemptJournal, CollectionStore
from .sampler import SAMPLER_VERSION, DeterministicSignalSampler, TaskDescriptor
from .sft_selection import (
    SFT_SELECTION_SCHEMA,
    load_sft_preflight_selection,
    validate_sft_preflight_selection,
)

__all__ = [
    "AppendOnlyAttemptJournal",
    "COLLECTION_SCHEMA",
    "CREDIT_FORMULA_VERSION",
    "CollectionStore",
    "DeterministicSignalSampler",
    "GROUP_SCHEMA",
    "RUN_IDENTITY_SCHEMA",
    "RunIdentity",
    "SAMPLER_VERSION",
    "SFT_SELECTION_SCHEMA",
    "TURN_SCHEMA",
    "TaskDescriptor",
    "assign_group_credit",
    "canonical_public_state",
    "load_sft_preflight_selection",
    "public_anchor_signature",
    "validate_committed_group",
    "validate_sft_preflight_selection",
    "validate_turn_evidence",
]
