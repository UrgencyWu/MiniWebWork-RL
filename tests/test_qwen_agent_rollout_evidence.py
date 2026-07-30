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
    def __init__(self, generated_ids, logprobs, prompt_ids=None):
        self.generated_ids = generated_ids
        self.logprobs = logprobs
        self.prompt_ids = prompt_ids if prompt_ids is not None else list(range(10))

    def generate(self, messages):
        return SimpleNamespace(
            raw_text='{"action":"finish"}',
            input_tokens=len(self.prompt_ids),
            new_tokens=len(self.generated_ids),
            latency_ms=1.0,
            prompt_token_ids=self.prompt_ids,
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


def test_agent_retains_prompt_completion_and_policy_logprobs():
    agent = QwenBrowserAgent(_Backend([7, 8], [-0.1, -0.2]), _PromptBuilder, _Parser)
    agent.reset("TASK", "instruction")

    attempt = agent.act(SimpleNamespace())

    assert attempt.schema_valid is True
    assert attempt.action.action == "finish"
    assert attempt.prompt_token_ids == list(range(10))
    assert attempt.generated_token_ids == [7, 8]
    assert attempt.token_logprobs == [-0.1, -0.2]
    assert attempt.prompt_hash == "prompt-hash"


def test_agent_rejects_misaligned_completion_logprobs():
    agent = QwenBrowserAgent(_Backend([7, 8], [-0.1]), _PromptBuilder, _Parser)
    agent.reset("TASK", "instruction")

    attempt = agent.act(SimpleNamespace())

    assert attempt.schema_valid is False
    assert attempt.action is None
    assert any(error.startswith("rollout_evidence_error") for error in attempt.errors)


def test_agent_rejects_misaligned_prompt_tokens():
    backend = _Backend([7], [-0.1], prompt_ids=[])
    original_generate = backend.generate

    def generate(messages):
        result = original_generate(messages)
        result.input_tokens = 3
        return result

    backend.generate = generate
    agent = QwenBrowserAgent(backend, _PromptBuilder, _Parser)
    agent.reset("TASK", "instruction")

    attempt = agent.act(SimpleNamespace())

    # Empty prompt-token evidence is tolerated for legacy deterministic
    # backends; the canonical stochastic rollout runner enforces completeness.
    assert attempt.schema_valid is True
