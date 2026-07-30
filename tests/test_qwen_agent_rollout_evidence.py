from types import SimpleNamespace

from miniwebwork.model_agent.qwen_agent import QwenBrowserAgent


class _PromptBuilder:
    @staticmethod
    def build_messages(observation, history):
        return [{"role": "user", "content": "test"}]

    @staticmethod
    def compute_message_hash(messages):
        return "prompt-hash"


class _Backend:
    def __init__(self, generated_ids, logprobs):
        self.generated_ids = generated_ids
        self.logprobs = logprobs

    def generate(self, messages):
        return SimpleNamespace(
            raw_text='{"action":"finish"}',
            input_tokens=10,
            new_tokens=len(self.generated_ids),
            latency_ms=1.0,
            generated_token_ids=self.generated_ids,
            logprobs=self.logprobs,
            sampling_logprobs=[],
            error="",
        )


class _Parser:
    @staticmethod
    def parse(raw_text):
        return SimpleNamespace(
            strict_json_success=True,
            fallback_used=False,
            parsed_payload={"action": "finish"},
            schema_valid=True,
            errors=[],
        )


def test_agent_retains_token_ids_and_policy_logprobs():
    agent = QwenBrowserAgent(_Backend([7, 8], [-0.1, -0.2]), _PromptBuilder, _Parser)
    agent.reset("TASK", "instruction")

    attempt = agent.act(SimpleNamespace())

    assert attempt.schema_valid is True
    assert attempt.action.action == "finish"
    assert attempt.generated_token_ids == [7, 8]
    assert attempt.token_logprobs == [-0.1, -0.2]
    assert attempt.prompt_hash == "prompt-hash"


def test_agent_rejects_misaligned_rollout_evidence():
    agent = QwenBrowserAgent(_Backend([7, 8], [-0.1]), _PromptBuilder, _Parser)
    agent.reset("TASK", "instruction")

    attempt = agent.act(SimpleNamespace())

    assert attempt.schema_valid is False
    assert attempt.action is None
    assert any(error.startswith("rollout_evidence_error") for error in attempt.errors)
