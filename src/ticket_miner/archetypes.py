"""Catalog of ticket archetypes.

Each archetype represents a real failure mode developers hit with the
OpenAI platform. The clustering pipeline does NOT see these labels;
they are kept on the synthetic data as ground truth for evaluating
cluster quality and for the report.

`weight_recent=True` archetypes get a timestamp distribution skewed
toward the last 7 days, so the severity scorer (volume × growth) has
something to detect.
"""

from __future__ import annotations

ARCHETYPES: dict[str, dict] = {
    "rate_limit": {
        "count": 70,
        "weight_recent": False,
        "summary": "HTTP 429 rate-limit errors (RPM/TPM)",
        "scenario": (
            "Customer is hitting OpenAI API rate limits. They see HTTP 429 "
            "responses with messages like 'Rate limit reached for gpt-4o in "
            "organization org-xxx on tokens per min (TPM): Limit 30000, Used "
            "30200, Requested 850'. They may not understand tier-based limits, "
            "retry-after headers, or how to implement exponential backoff. "
            "Often triggered by parallel batch jobs or sudden traffic spikes."
        ),
        "code_pattern": (
            "asyncio.gather of many client.chat.completions.create calls with "
            "no semaphore or backoff"
        ),
    },
    "context_overflow": {
        "count": 55,
        "weight_recent": False,
        "summary": "Context length exceeded for long inputs",
        "scenario": (
            "Customer sending a long document or growing chat history and "
            "getting BadRequestError: 'This model's maximum context length is "
            "128000 tokens. However, your messages resulted in 142000 tokens.' "
            "They may not know about tiktoken, or how to chunk/summarize. "
            "Sometimes their conversation history grows unbounded over turns."
        ),
        "code_pattern": "appending to messages list across many turns without trimming",
    },
    "tool_schema_mismatch": {
        "count": 50,
        "weight_recent": True,  # GROWING — severity scorer should flag this
        "summary": "Tool/function-calling schema validation errors",
        "scenario": (
            "Customer using function calling or tool use and either (a) the "
            "model returns arguments that don't match their JSON schema, or "
            "(b) they get an error like 'Invalid schema for function: "
            '\\"parameters\\" must contain at least one property when '
            '\\"type\\" is \\"object\\".\' Confusion around strict mode, '
            "additionalProperties:false, required arrays, and $ref usage."
        ),
        "code_pattern": (
            "tools=[{...}] with a malformed JSON Schema (missing required, "
            "wrong additionalProperties, or strict=True with $refs)"
        ),
    },
    "streaming_disconnect": {
        "count": 45,
        "weight_recent": False,
        "summary": "Streaming responses dropping mid-generation",
        "scenario": (
            "Customer using stream=True and the connection drops mid-response, "
            "or they see APIConnectionError / IncompleteRead. Could be load "
            "balancer timeouts, async context cancellation, or proxy "
            "buffering. Often happens with long generations on slower models "
            "or when their server has aggressive idle timeouts."
        ),
        "code_pattern": "async for chunk in client.chat.completions.create(stream=True)",
    },
    "auth_error": {
        "count": 40,
        "weight_recent": False,
        "summary": "401 unauthorized / wrong key / org-project mismatch",
        "scenario": (
            "Customer hitting AuthenticationError 401. Common causes: revoked "
            "key, wrong org or project ID, expired key, key from a different "
            "account, copy-paste errors with whitespace, or using a project "
            "key without specifying the project."
        ),
        "code_pattern": "OpenAI(api_key=...) with mismatched org or stale key",
    },
    "billing_quota": {
        "count": 40,
        "weight_recent": False,
        "summary": "Billing limits / quota exhausted / payment failures",
        "scenario": (
            "Customer hit their monthly spend limit, has a failed credit "
            "card, or sees 'You exceeded your current quota'. Distinct from "
            "rate limits — this is account-level billing, not per-minute. "
            "Free/Plus users sometimes confuse ChatGPT vs API billing."
        ),
        "code_pattern": "n/a",
    },
    "structured_output_invalid": {
        "count": 40,
        "weight_recent": True,  # GROWING
        "summary": "JSON mode / structured outputs returning invalid output",
        "scenario": (
            "Customer using response_format json_object or json_schema and "
            "the model returns truncated JSON, JSON that doesn't match their "
            "schema, or the request errors with 'response_format must contain "
            "a JSON Schema with additionalProperties: false on every object'."
        ),
        "code_pattern": "response_format={'type': 'json_schema', 'json_schema': {...}}",
    },
    "batch_api_stuck": {
        "count": 35,
        "weight_recent": False,
        "summary": "Batch API jobs not completing or stuck in 'in_progress'",
        "scenario": (
            "Customer submitted a Batch API job that's been in 'in_progress' "
            "or 'validating' for many hours. Or it failed with 'Token rate "
            "limit exceeded' at the batch level. Or they can't find the "
            "output file once it completed."
        ),
        "code_pattern": "client.batches.create(input_file_id=...)",
    },
    "fine_tuning_failure": {
        "count": 30,
        "weight_recent": False,
        "summary": "Fine-tuning jobs failing on dataset format or hyperparameters",
        "scenario": (
            "Fine-tuning job failed during validation or training. Common "
            "causes: malformed JSONL, missing 'messages' key, role values "
            "wrong, exceeding tokens-per-example limit, or hyperparameter "
            "values out of allowed range."
        ),
        "code_pattern": "client.fine_tuning.jobs.create(training_file=..., model=...)",
    },
    "image_input_error": {
        "count": 25,
        "weight_recent": False,
        "summary": "Vision/image input errors (size, format, URL access)",
        "scenario": (
            "Customer sending image inputs (base64 or URL) and getting "
            "errors: image too large, unsupported format, URL not "
            "accessible from OpenAI's side, or the model isn't extracting "
            "the information they expect from the image."
        ),
        "code_pattern": "messages with image_url content blocks",
    },
    "noise": {
        "count": 70,
        "weight_recent": False,
        "summary": "General questions, off-topic, account housekeeping",
        "scenario": (
            "Generic questions that should NOT cluster with technical bug "
            "clusters: How do I cancel ChatGPT Plus? Is feature X available "
            "yet? Pricing questions for a new project. Where do I find my "
            "org ID? Lost password. These should appear as small clusters "
            "or noise points in HDBSCAN."
        ),
        "code_pattern": "n/a",
    },
}


def total_base_count() -> int:
    return sum(cfg["count"] for cfg in ARCHETYPES.values())
