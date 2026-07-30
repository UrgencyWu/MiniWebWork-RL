from miniwebwork.model_agent.failure_analysis import classify_failures
from miniwebwork.model_agent.metrics import compute_metrics


def test_infrastructure_failure_is_excluded_from_policy_denominators():
    results = [
        {
            "task_id": "OK",
            "success": True,
            "reward": 1.0,
            "rollout_valid": True,
            "failure_origin": "none",
            "model_turns": 1,
            "environment_steps": 1,
            "termination_reason": "verified_submission",
            "turns": [
                {
                    "raw_output": '{"action":"finish"}',
                    "strict_json_success": True,
                    "fallback_used": False,
                    "schema_valid": True,
                    "action": {"action": "finish"},
                    "action_result": {"success": True},
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "latency_ms": 1,
                }
            ],
        },
        {
            "task_id": "INFRA",
            "success": False,
            "reward": None,
            "rollout_valid": False,
            "failure_origin": "infrastructure",
            "model_turns": 0,
            "environment_steps": 0,
            "termination_reason": "model_backend_error",
            "turns": [],
        },
    ]

    metrics = compute_metrics(results)

    assert metrics["requested_tasks"] == 2
    assert metrics["valid_tasks"] == 1
    assert metrics["infrastructure_errors"] == 1
    assert metrics["success_rate"] == 1.0
    assert metrics["total_reward"] == 1.0
    assert metrics["strict_json_success_rate"] == 1.0


def test_failure_analysis_keeps_infrastructure_separate():
    analysis = classify_failures(
        [
            {
                "task_id": "INFRA",
                "success": False,
                "rollout_valid": False,
                "failure_origin": "infrastructure",
                "termination_reason": "environment_step_error",
                "error": "browser died",
            },
            {
                "task_id": "POLICY",
                "success": False,
                "rollout_valid": True,
                "failure_origin": "policy",
                "termination_reason": "premature_finish",
                "turns": [
                    {
                        "strict_json_success": True,
                        "schema_valid": True,
                        "action": {"action": "finish"},
                        "action_result": {"success": True},
                    }
                ],
                "failure_reasons": ["missing_submission"],
            },
        ]
    )

    assert analysis["summary"]["infrastructure_failures"] == 1
    assert analysis["summary"]["policy_failures"] == 1
    by_task = {item["task_id"]: item for item in analysis["analyzed_tasks"]}
    assert by_task["INFRA"]["primary_failure"] == "infrastructure_failure"
    assert by_task["POLICY"]["primary_failure"] == "premature_finish"
