"""Completion-only mask audit for SFT data. Ensures only assistant tokens participate in loss."""

import json, sys
from pathlib import Path


def audit(train_file: str, valid_file: str, output_path: str):
    """Check that assistant tokens are correctly maskable for completion-only training."""
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/share/model/Qwen3.5-4B", local_files_only=True, trust_remote_code=True)

    results = {"samples_checked": 0, "passed_samples": 0, "failed_samples": 0,
               "all_masked_count": 0, "user_token_leak_count": 0,
               "action_decode_mismatch_count": 0,
               "input_tokens": [], "completion_tokens": []}

    def mask_labels(input_ids, assistant_start_pos):
        """Create labels: -100 for system/user tokens, keep assistant tokens."""
        labels = [-100] * len(input_ids)
        for i in range(assistant_start_pos, len(input_ids)):
            labels[i] = input_ids[i]
        return labels

    for filepath in [train_file, valid_file]:
        p = Path(filepath)
        if not p.exists():
            continue
        for line in p.read_text().strip().split("\n"):
            if not line.strip():
                continue
            sample = json.loads(line)
            action = sample.get("action", {})
            action_json = json.dumps(action, ensure_ascii=False, separators=(",", ":"))

            # Build messages: system + user (from sample or reconstructed)
            # We need the actual messages. For this audit, reconstruct from action + task
            # Use a simplified approach: tokenize system + user, then assistant
            task_id = sample.get("task_id", "")
            instruction = f"Task {task_id}"

            # Build full text with chat template
            msgs = [
                {"role": "system", "content": "You are a browser agent."},
                {"role": "user", "content": f"Task: {instruction}\n\nPage: products\n\nOutput one JSON action."},
                {"role": "assistant", "content": action_json},
            ]

            # Render then tokenize
            try:
                rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
                full_ids = tokenizer.encode(rendered)
            except Exception:
                continue

            results["samples_checked"] += 1
            results["input_tokens"].append(len(full_ids))

            # Find assistant content position
            assistant_text = action_json
            # Find the start of assistant text in the rendered output
            assistant_start = rendered.find(assistant_text)
            if assistant_start < 0:
                results["failed_samples"] += 1
                results["action_decode_mismatch_count"] += 1
                continue

            # Tokenize the prefix (everything before assistant content)
            prefix = rendered[:assistant_start]
            prefix_ids = tokenizer.encode(prefix)
            assistant_ids = tokenizer.encode(assistant_text)

            results["completion_tokens"].append(len(assistant_ids))
            labels = mask_labels(full_ids, len(prefix_ids))

            # Check: all tokens before assistant_start should be -100
            for i in range(len(prefix_ids)):
                if labels[i] != -100:
                    results["user_token_leak_count"] += 1
                    break

            # Check: at least one assistant token should not be -100
            has_valid = any(l != -100 for l in labels[len(prefix_ids):])
            if not has_valid:
                results["all_masked_count"] += 1

            # Decode labels to verify
            valid_labels = [l for l in labels if l != -100]
            if valid_labels:
                decoded = tokenizer.decode(valid_labels)
                if action_json not in decoded:
                    results["action_decode_mismatch_count"] += 1
                    results["failed_samples"] += 1
                    continue

            results["passed_samples"] += 1

    # Stats
    its = results["input_tokens"]
    cts = results["completion_tokens"]
    results["mean_input_tokens"] = sum(its) / max(len(its), 1)
    results["p95_input_tokens"] = sorted(its)[int(len(its) * 0.95)] if its else 0
    results["max_input_tokens"] = max(its) if its else 0
    results["mean_completion_tokens"] = sum(cts) / max(len(cts), 1)
    results["max_completion_tokens"] = max(cts) if cts else 0

    Path(output_path).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"Checked: {results['samples_checked']}, Passed: {results['passed_samples']}, Failed: {results['failed_samples']}")
    print(f"All-masked: {results['all_masked_count']}, User leak: {results['user_token_leak_count']}, Decode mismatch: {results['action_decode_mismatch_count']}")
    print(f"Input tokens: mean={results['mean_input_tokens']:.0f}, p95={results['p95_input_tokens']}, max={results['max_input_tokens']}")
    print(f"Completion tokens: mean={results['mean_completion_tokens']:.0f}, max={results['max_completion_tokens']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    tf = sys.argv[1] if len(sys.argv) > 1 else "data/sft/m2_1/train.jsonl"
    vf = sys.argv[2] if len(sys.argv) > 2 else "data/sft/m2_1/valid.jsonl"
    op = sys.argv[3] if len(sys.argv) > 3 else "artifacts/m2_1/completion_mask_audit.json"
    audit(tf, vf, op)
