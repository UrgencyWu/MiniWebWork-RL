from types import SimpleNamespace

from miniwebwork.agent_env.schemas import Observation
from miniwebwork.model_agent.agent_loop import run_model_episode


class _Environment:
    def __init__(self):
        self.step_calls = 0
        self.trajectory = None

    def reset(self, task_id):
        return Observation(
            task_id=task_id,
            episode_id="EP",
            instruction="instruction",
            page_type="products",
        )

    def set_agent_name(self, name):
        self.agent_name = name

    def step(self, action):
        self.step_calls += 1
        raise AssertionError("env.step must not be called for invalid output")


class _InvalidAgent:
    def __init__(self, backend_error=False):
        self._turn = 0
        self.feedback_calls = 0
        self.backend_error = backend_error

    def reset(self, task_id, instruction):
        self._turn = 0

    @property
    def model_turn(self):
        return self._turn

    def act(self, observation):
        self._turn += 1
        errors = ["generation_error: CUDA"] if self.backend_error else ["unknown_action"]
        return SimpleNamespace(
            model_turn_index=self._turn,
            prompt_hash="hash",
            prompt_token_ids=[1, 2],
            input_tokens=2,
            raw_output="bad",
            generated_token_ids=[3],
            token_logprobs=[-1.0],
            sampling_logprobs=[-1.0],
            output_tokens=1,
            latency_ms=1.0,
            strict_json_success=False,
            fallback_used=False,
            schema_valid=False,
            action=None,
            errors=errors,
        )

    def record_feedback(self, attempt, action_result, page_type):
        self.feedback_calls += 1


def test_schema_invalid_turns_do_not_enter_executed_history():
    environment = _Environment()
    agent = _InvalidAgent()

    result = run_model_episode("TASK", environment, agent)

    assert result["rollout_valid"] is True
    assert result["reward"] == 0.0
    assert result["termination_reason"] == "model_output_failure_limit"
    assert result["model_turns"] == 3
    assert environment.step_calls == 0
    assert agent.feedback_calls == 0


def test_backend_error_is_infrastructure_not_policy_failure():
    environment = _Environment()
    agent = _InvalidAgent(backend_error=True)

    result = run_model_episode("TASK", environment, agent)

    assert result["success"] is False
    assert result["rollout_valid"] is False
    assert result["failure_origin"] == "infrastructure"
    assert result["reward"] is None
    assert result["termination_reason"] == "model_backend_error"
    assert environment.step_calls == 0
