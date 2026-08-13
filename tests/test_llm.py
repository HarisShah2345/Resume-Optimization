"""Tests for the `tools.llm` transport layer — structured-output decoding.

The agent tools are tested with mocks elsewhere; this file exercises the REAL
`structured_call` code against a fake `client.messages.stream`, proving the
two structured-output mechanisms end-to-end without a live API:

  - GATEWAY path: forced tool-use constrained decoding (`tools` +
    `tool_choice`), tool_use.input → `model_validate`, partial-JSON streaming,
    thinking-only recovery, and transient-5xx retry with backoff.
  - NATIVE path: `output_format` (Pydantic) → `parsed_output`, with
    `cache_control` correctly nested INSIDE the system content block.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import anthropic
import httpx
import pytest
from pydantic import BaseModel

import config
import tools.llm as llm
from tools.llm import structured_call


@pytest.fixture(autouse=True)
def _reset_native_schema_memo(tmp_path, monkeypatch):
    """`_native_schema_failures` / `_persisted_schema_failures` are
    process-global; a schema rejected in one test must not make a later
    test's native call skip native (tests are independent). Clear both before
    and after every test, and redirect the persisted-cache file to a
    throwaway tmp dir so tests never read or write the real on-disk cache."""
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    llm._native_schema_failures.clear()
    llm._persisted_schema_failures.clear()
    yield
    llm._native_schema_failures.clear()
    llm._persisted_schema_failures.clear()


class Sample(BaseModel):
    name: str
    count: int


# --- fake SDK pieces ---------------------------------------------------------
def _thinking_block():
    return SimpleNamespace(type="thinking", thinking="burning budget...")


def _tool_use_block(payload: dict):
    return SimpleNamespace(type="tool_use", name="structured_output", input=payload)


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _final(content, parsed_output=None, stop_reason="end_turn"):
    return SimpleNamespace(content=content, parsed_output=parsed_output, stop_reason=stop_reason)


def _json_delta(chunk: str):
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(partial_json=chunk))


def _text_delta(chunk: str):
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text=chunk))


class FakeStream:
    """Stand-in for `client.messages.stream(...)` context manager."""

    def __init__(self, final, raw_events=None):
        self._final = final
        self._events = raw_events or []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield from self._events

    @property
    def text_stream(self):
        for e in self._events:
            if getattr(e, "delta", None) is not None and getattr(e.delta, "text", None):
                yield e.delta.text

    def get_final_message(self):
        return self._final


class FakeMessages:
    """Replays a script of [stream | exception] items, recording requests."""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []
        self.stream_count = 0

    def stream(self, **kwargs):
        self.requests.append(kwargs)
        item = self.script[min(self.stream_count, len(self.script) - 1)]
        self.stream_count += 1
        if isinstance(item, BaseException):
            raise item
        return item


class _Transient(Exception):
    pass


def _gateway(monkeypatch):
    monkeypatch.setattr(config, "API_BASE_URL", "http://gateway.test")


def _fake_client(messages):
    return patch("tools.llm._client", return_value=SimpleNamespace(messages=messages))


# --- gateway path ------------------------------------------------------------
def test_gateway_path_uses_tool_use_decoding(monkeypatch):
    _gateway(monkeypatch)
    payload = {"name": "Jane", "count": 3}
    fake = FakeMessages([FakeStream(_final([_tool_use_block(payload)]))])
    with _fake_client(fake):
        result = structured_call("free-bundle", "sys", "user", Sample, 16000)

    assert result == Sample(name="Jane", count=3)
    req = fake.requests[0]
    assert req["model"] == "free-bundle"
    assert req["tools"][0]["name"] == "structured_output"
    assert req["tool_choice"] == {"type": "tool", "name": "structured_output"}
    # cache_control lives inside the system block, never top-level.
    assert "cache_control" not in req
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_gateway_string_typed_tool_input(monkeypatch):
    """Some gateways return tool_use.input as a JSON STRING instead of a parsed
    dict (verified live on free-bundle: the whole ResumeContent arrived as
    `input='[...]'`). Must validate as JSON, not as a Python dict."""
    _gateway(monkeypatch)
    fake = FakeMessages(
        [FakeStream(_final([_tool_use_block('{"name": "Jane", "count": 3}')]))]
    )
    with _fake_client(fake):
        result = structured_call("free-bundle", "sys", "user", Sample, 16000)
    assert result == Sample(name="Jane", count=3)


def test_thinking_only_response_recovers_without_thinking(monkeypatch):
    """A combo model that burns its whole budget on thinking returns no
    tool_use → the call retries once with thinking off and a larger budget."""
    _gateway(monkeypatch)
    payload = {"name": "Jane", "count": 3}
    fake = FakeMessages(
        [
            FakeStream(_final([_thinking_block()], stop_reason="max_tokens")),
            FakeStream(_final([_tool_use_block(payload)])),
        ]
    )
    with _fake_client(fake):
        result = structured_call("free-bundle", "sys", "user", Sample, 16000, thinking=True)

    assert result == Sample(name="Jane", count=3)
    assert fake.stream_count == 2
    assert fake.requests[0].get("thinking") == {"type": "adaptive"}
    # Retried WITHOUT thinking AND with a 4x budget bump.
    assert "thinking" not in fake.requests[1]
    assert fake.requests[1]["max_tokens"] == 16000 * 4


def test_thinking_only_at_max_budget_still_retries_once(monkeypatch):
    """Even when the budget is already at the 131072 ceiling (as after the
    thinking-only recovery bump), a thinking-only response still gets exactly
    ONE retry — and a second thinking-only response fails loudly instead of
    recursing forever."""
    _gateway(monkeypatch)
    payload = {"name": "Jane", "count": 3}
    fake = FakeMessages(
        [
            FakeStream(_final([_thinking_block()], stop_reason="max_tokens")),
            FakeStream(_final([_thinking_block()], stop_reason="max_tokens")),
        ]
    )
    with _fake_client(fake):
        try:
            structured_call(
                "free-bundle", "sys", "user", Sample, config.GATEWAY_MAX_TOKENS
            )
        except RuntimeError as exc:
            assert "no structured_output tool_use block" in str(exc)
        else:
            raise AssertionError("expected RuntimeError after the single retry")
    assert fake.stream_count == 2  # bounded: exactly one retry
    assert "thinking" not in fake.requests[1]


def test_autothink_without_request_recovers_with_bigger_budget(monkeypatch):
    """Combo gateways auto-think even when thinking is NOT requested — a
    thinking-only, budget-exhausted response must still recover by bumping
    the budget (the exact failure that hit the live e2e run)."""
    _gateway(monkeypatch)
    payload = {"name": "Jane", "count": 3}
    fake = FakeMessages(
        [
            FakeStream(_final([_thinking_block()], stop_reason="max_tokens")),
            FakeStream(_final([_tool_use_block(payload)])),
        ]
    )
    with _fake_client(fake):
        result = structured_call("free-bundle", "sys", "user", Sample, 16000, thinking=False)

    assert result == Sample(name="Jane", count=3)
    assert fake.stream_count == 2
    assert "thinking" not in fake.requests[0]   # we never asked for it
    assert "thinking" not in fake.requests[1]
    assert fake.requests[1]["max_tokens"] == 16000 * 4


def test_text_json_fallback(monkeypatch):
    """Some gateways answer with bare JSON in text instead of a tool_use block."""
    _gateway(monkeypatch)
    fake = FakeMessages(
        [FakeStream(_final([_text_block('Here you go: {"name": "Jane", "count": 3}')]))]
    )
    with _fake_client(fake):
        result = structured_call("free-bundle", "sys", "user", Sample, 16000)
    assert result == Sample(name="Jane", count=3)


def _bad_request(msg: str) -> anthropic.BadRequestError:
    """A real SDK 400 error (the free-bundle gateway's schema-complexity
    rejection is exactly this shape)."""
    request = httpx.Request("POST", "http://gateway.test/v1/messages")
    response = httpx.Response(
        400,
        request=request,
        headers={"request-id": "req_test"},
        json={"type": "error", "error": {"type": "invalid_request_error", "message": msg}},
    )
    return anthropic.BadRequestError(msg, response=response, body=response.json())


def test_schema_too_complex_falls_back_to_text_json(monkeypatch):
    """A 400 'Schema is too complex.' (the real-API limit the free-bundle
    gateway surfaces for the full ResumeContent schema) must NOT fail the run:
    it re-issues the call with NO tools and validates the text JSON instead."""
    _gateway(monkeypatch)
    monkeypatch.setattr("tools.llm.time.sleep", lambda _s: None)
    payload = '{"name": "Jane", "count": 3}'
    fake = FakeMessages(
        [
            _bad_request("Schema is too complex."),
            FakeStream(_final([_text_block(payload)])),
        ]
    )
    with _fake_client(fake):
        result = structured_call("free-bundle", "sys", "user", Sample, 16000)

    assert result == Sample(name="Jane", count=3)
    assert fake.stream_count == 2
    # The fallback request has no tools/tool_choice — that's what avoids the limit.
    assert "tools" not in fake.requests[1]
    assert "tool_choice" not in fake.requests[1]
    # ...and it carries the field-typed schema skeleton so the model stays on-shape.
    content = fake.requests[1]["messages"][0]["content"]
    assert "## required JSON shape" in content
    assert content.endswith("- count: integer")


def test_schema_too_complex_streams_fallback_json_live(monkeypatch):
    """The text-JSON fallback still streams deltas to on_token on success."""
    _gateway(monkeypatch)
    monkeypatch.setattr("tools.llm.time.sleep", lambda _s: None)
    seen: list[str] = []
    fake = FakeMessages(
        [
            _bad_request("Schema is too complex."),
            FakeStream(
                _final([_text_block('{"name": "Jane", "count": 3}')]),
                # The fallback has no tools, so deltas are plain text, not partial_json.
                [_text_delta('{"name": "Ja'), _text_delta('ne", "count": 3}')],
            ),
        ]
    )
    with _fake_client(fake):
        structured_call("free-bundle", "sys", "user", Sample, 16000, on_token=seen.append)
    assert "".join(seen) == '{"name": "Jane", "count": 3}'


def test_gateway_grammar_timeout_falls_back_to_text_json(monkeypatch):
    """Gateways can reject the tool-use schema with `400 Grammar compilation
    timed out` too — same schema-less recovery as the 'too complex' variant."""
    _gateway(monkeypatch)
    monkeypatch.setattr("tools.llm.time.sleep", lambda _s: None)
    payload = '{"name": "Jane", "count": 3}'
    fake = FakeMessages(
        [
            _bad_request("Grammar compilation timed out"),
            FakeStream(_final([_text_block(payload)])),
        ]
    )
    with _fake_client(fake):
        result = structured_call("free-bundle", "sys", "user", Sample, 16000)

    assert result == Sample(name="Jane", count=3)
    assert fake.stream_count == 2
    assert "tools" not in fake.requests[1]
    assert "tool_choice" not in fake.requests[1]


def test_non_schema_400_propagates(monkeypatch):
    """Only schema-complexity 400s are absorbed; any other 400 still fails the
    call loudly instead of being masked as a schema problem."""
    _gateway(monkeypatch)
    fake = FakeMessages([_bad_request("Model does not exist")])
    with _fake_client(fake):
        try:
            structured_call("free-bundle", "sys", "user", Sample, 16000)
        except anthropic.BadRequestError as exc:
            assert "too complex" not in str(exc).lower()
        else:
            raise AssertionError("expected the non-schema 400 to propagate")
    assert fake.stream_count == 1  # no retry, no fallback


def test_transient_failure_is_retried_and_streams_only_success(monkeypatch):
    """A 5xx/pool-exhaustion failure must be retried with backoff; the partial
    deltas of the failed attempt must NOT reach the UI."""
    _gateway(monkeypatch)
    monkeypatch.setattr("tools.llm._RETRYABLE", (_Transient,))
    monkeypatch.setattr("tools.llm.time.sleep", lambda _s: None)
    payload = {"name": "Jane", "count": 3}
    seen: list[str] = []
    fake = FakeMessages(
        [
            _Transient("pool exhausted"),
            FakeStream(
                _final([_tool_use_block(payload)]),
                [_json_delta('{"name": "Ja'), _json_delta('ne", "count": 3}')],
            ),
        ]
    )
    with _fake_client(fake):
        result = structured_call("free-bundle", "sys", "user", Sample, 16000, on_token=seen.append)

    assert result == Sample(name="Jane", count=3)
    assert fake.stream_count == 2
    # Only the successful attempt's deltas arrive, in order.
    assert "".join(seen) == '{"name": "Jane", "count": 3}'


def test_wall_clock_timeout_retries_then_raises(monkeypatch):
    """A stream that hangs (no delta, no error) must not block the agent
    forever: the per-attempt deadline fails the attempt and, after all retries
    hang too, the call raises loudly. This is the exact failure that froze the
    live e2e for 30+ minutes on free-bundle."""
    _gateway(monkeypatch)
    monkeypatch.setattr("tools.llm._ATTEMPT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("tools.llm.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def _hung(cb):
        calls["n"] += 1
        # Block WITHOUT time.sleep — the backoff patch above also patches the
        # global time module, so a sleep here would be a no-op and never hang.
        threading.Event().wait(0.5)  # worker thread blocks past the deadline

    try:
        llm._call_with_retry(_hung, None)
    except TimeoutError as exc:
        assert "wall-clock deadline" in str(exc)
    else:
        raise AssertionError("expected TimeoutError after every attempt hung")
    assert calls["n"] == 3  # bounded: exactly _MAX_ATTEMPTS, then failure


def test_wall_clock_timeout_then_success(monkeypatch):
    """One hung attempt should not fail the call if a later attempt succeeds —
    the deadline only abandons the stalled attempt, the next one runs fresh."""
    _gateway(monkeypatch)
    monkeypatch.setattr("tools.llm._ATTEMPT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("tools.llm.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def _flaky(cb):
        calls["n"] += 1
        if calls["n"] == 1:
            threading.Event().wait(0.5)  # first attempt hangs past the deadline
        return "done"

    assert llm._call_with_retry(_flaky, None) == "done"
    assert calls["n"] == 2


def test_gateway_streams_partial_json_live(monkeypatch):
    """Happy path: the tool-arg JSON streams to on_token on the first attempt."""
    _gateway(monkeypatch)
    payload = {"name": "Jane", "count": 3}
    seen: list[str] = []
    fake = FakeMessages(
        [
            FakeStream(
                _final([_tool_use_block(payload)]),
                [_json_delta('{"na'), _json_delta('me": "Jane", "count": 3}')],
            )
        ]
    )
    with _fake_client(fake):
        structured_call("free-bundle", "sys", "user", Sample, 16000, on_token=seen.append)
    assert "".join(seen) == '{"name": "Jane", "count": 3}'


# --- native path -------------------------------------------------------------
def test_native_path_uses_output_format(monkeypatch):
    monkeypatch.setattr(config, "API_BASE_URL", None)  # direct Anthropic API
    parsed = Sample(name="Jane", count=3)
    fake = FakeMessages([FakeStream(_final([_text_block("{}")], parsed_output=parsed))])
    with _fake_client(fake):
        result = structured_call("claude-sonnet-5", "sys", "user", Sample, 16000)

    assert result is parsed
    req = fake.requests[0]
    assert req["output_format"] is Sample
    assert "tools" not in req
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_native_schema_too_complex_falls_back_to_text_json(monkeypatch):
    """The direct API's native `output_format` rejects the full ResumeContent
    schema (`400 Schema is too complex.` — verified live). The tool-use path
    shares the same grammar machinery, so the native path skips straight to the
    schema-less text-JSON recovery instead of failing the run."""
    monkeypatch.setattr(config, "API_BASE_URL", None)  # direct Anthropic API
    monkeypatch.setattr("tools.llm.time.sleep", lambda _s: None)
    payload = '{"name": "Jane", "count": 3}'
    fake = FakeMessages(
        [
            _bad_request("Schema is too complex."),
            FakeStream(_final([_text_block(payload)])),
        ]
    )
    with _fake_client(fake):
        result = structured_call("claude-sonnet-5", "sys", "user", Sample, 16000)

    assert result == Sample(name="Jane", count=3)
    assert fake.stream_count == 2
    # The recovery request drops output_format AND tools — no grammar at all.
    assert "output_format" not in fake.requests[1]
    assert "tools" not in fake.requests[1]
    assert "tool_choice" not in fake.requests[1]


def test_native_grammar_timeout_falls_back_to_text_json(monkeypatch):
    """The direct API can also reject a big schema with `400 Grammar compilation
    timed out` (verified live — the server spends minutes compiling the grammar,
    then 400s). The same schema-less recovery must apply."""
    monkeypatch.setattr(config, "API_BASE_URL", None)  # direct Anthropic API
    monkeypatch.setattr("tools.llm.time.sleep", lambda _s: None)
    payload = '{"name": "Jane", "count": 3}'
    fake = FakeMessages(
        [
            _bad_request("Grammar compilation timed out"),
            FakeStream(_final([_text_block(payload)])),
        ]
    )
    with _fake_client(fake):
        result = structured_call("claude-sonnet-5", "sys", "user", Sample, 16000)

    assert result == Sample(name="Jane", count=3)
    assert fake.stream_count == 2
    assert "output_format" not in fake.requests[1]
    assert "tools" not in fake.requests[1]


def test_native_schema_rejection_is_memoized_for_followup_calls(monkeypatch):
    """Once a schema is known to 400 on the direct API's native `output_format`,
    every later call for the SAME schema in this process must skip the doomed
    native attempt and go straight to text-JSON — otherwise each repair/validate
    pass in one agent run re-pays ~3 minutes of grammar-compile rejection."""
    monkeypatch.setattr(config, "API_BASE_URL", None)  # direct Anthropic API
    monkeypatch.setattr("tools.llm.time.sleep", lambda _s: None)
    payload = '{"name": "Jane", "count": 3}'
    fake = FakeMessages(
        [
            _bad_request("Schema is too complex."),   # 1st call: native 400 → memo + text-json
            FakeStream(_final([_text_block(payload)])),
            FakeStream(_final([_text_block(payload)])),  # 2nd call: native skipped entirely
        ]
    )
    with _fake_client(fake):
        r1 = structured_call("claude-sonnet-5", "sys", "user", Sample, 16000)
        r2 = structured_call("claude-sonnet-5", "sys", "user", Sample, 16000)

    assert r1 == Sample(name="Jane", count=3)
    assert r2 == Sample(name="Jane", count=3)
    # request[0] = doomed native attempt; request[1] = text-json (call 1);
    # request[2] = text-json (call 2, native skipped, so no third native attempt).
    assert len(fake.requests) == 3
    assert "output_format" in fake.requests[0]
    assert "output_format" not in fake.requests[1]
    assert "output_format" not in fake.requests[2]
    assert "tools" not in fake.requests[2]


# --- nested double-encoding (bug surfaced by the live e2e) -------------------
class PayloadContact(BaseModel):
    name: str
    phone: str


class Payload(BaseModel):
    name: str
    contact: PayloadContact
    year: str
    tags: list[str]


def test_gateway_decodes_double_encoded_nested_fields(monkeypatch):
    """Surfaced by the live e2e: some gateways double-encode nested structured
    fields (the `contact` object of ResumeContent arrived as a JSON string).
    They must be decoded recursively before validation, while scalar-looking
    strings (dates, names) stay untouched."""
    _gateway(monkeypatch)
    payload = {
        "name": "Jane",
        "contact": '{"name": "Jane", "phone": "555"}',  # nested model as a JSON string
        "year": "2015",                                  # scalar string left alone
        "tags": '["a", "b"]',                            # list as a JSON string
    }
    fake = FakeMessages([FakeStream(_final([_tool_use_block(payload)]))])
    with _fake_client(fake):
        result = structured_call("free-bundle", "sys", "user", Payload, 16000)

    assert result == Payload(
        name="Jane",
        contact=PayloadContact(name="Jane", phone="555"),
        year="2015",
        tags=["a", "b"],
    )


def test_gateway_decodes_double_encoded_top_level_string(monkeypatch):
    """The same fix must apply when the ENTIRE payload arrives as a JSON string
    that itself contains nested double-encoded fields."""
    _gateway(monkeypatch)
    payload = (
        '{"name": "Jane", "contact": "{\\"name\\": \\"Jane\\", \\"phone\\": \\"555\\"}", '
        '"year": "2015", "tags": "[]"}'
    )
    fake = FakeMessages([FakeStream(_final([_tool_use_block(payload)]))])
    with _fake_client(fake):
        result = structured_call("free-bundle", "sys", "user", Payload, 16000)

    assert result == Payload(
        name="Jane",
        contact=PayloadContact(name="Jane", phone="555"),
        year="2015",
        tags=[],
    )


def test_decode_nested_json_recurses_and_preserves_scalars():
    """Helper-level edge cases: dicts/lists decode at any depth; strings that
    merely look braced but are not valid JSON, and plain/scalar strings, are
    returned unchanged."""
    d = llm._decode_nested_json
    assert d({"contact": '{"name": "Jane"}'}) == {"contact": {"name": "Jane"}}
    assert d('{"name": "Jane", "count": 3}') == {"name": "Jane", "count": 3}
    assert d(["x", '{"b": 2}']) == ["x", {"b": 2}]
    assert d({"contact": '{"meta": "{\\"deep\\": 1}"}'}) == {"contact": {"meta": {"deep": 1}}}
    assert d("2015") == "2015"
    assert d("{not json}") == "{not json}"
    assert d("just plain text") == "just plain text"


# --- schema skeleton (text-JSON shape hint, bug surfaced by the live e2e) ----
def test_schema_skeleton_renders_flat_field_types():
    s = llm._schema_skeleton(Sample)
    assert "- name: string" in s
    assert "- count: integer" in s


def test_schema_skeleton_expands_nested_models_and_arrays():
    """Nested models and arrays must be expanded inline so the model emits the
    right shape — the exact failure on the direct API: ResumeContent `skills`
    came back as a flat string list instead of [{category, items}]."""
    s = llm._schema_skeleton(Payload)
    assert "- name: string" in s
    assert "- contact: {name: string, phone: string}" in s
    assert "- tags: array of string" in s
    assert "- year: string" in s


def test_schema_skeleton_prompt_is_appended_by_text_json(monkeypatch):
    """The text-JSON fallback prompt must carry the skeleton so the free-text
    model sees the schema shape it is otherwise blind to."""
    _gateway(monkeypatch)
    monkeypatch.setattr("tools.llm.time.sleep", lambda _s: None)
    payload = '{"name": "Jane", "count": 3}'
    fake = FakeMessages(
        [
            _bad_request("Schema is too complex."),
            FakeStream(_final([_text_block(payload)])),
        ]
    )
    with _fake_client(fake):
        structured_call("free-bundle", "sys", "user", Sample, 16000)
    content = fake.requests[1]["messages"][0]["content"]
    assert "- name: string" in content
    assert "- count: integer" in content


# --- text-JSON thinking-exhaustion retry (bug surfaced by the live e2e) ------
def test_text_json_thinking_exhaustion_retries_with_bumped_budget(monkeypatch):
    """A direct-API model can auto-think even when thinking isn't requested, and
    a hard task can burn the whole budget on thinking before emitting the JSON
    (verified live: content=['thinking'], stop_reason=max_tokens on the repair
    pass). The text-JSON path must retry once with a bumped budget."""
    monkeypatch.setattr(config, "API_BASE_URL", None)  # direct Anthropic API
    monkeypatch.setattr("tools.llm.time.sleep", lambda _s: None)
    payload = '{"name": "Jane", "count": 3}'
    fake = FakeMessages(
        [
            _bad_request("Schema is too complex."),
            FakeStream(_final([_thinking_block()], stop_reason="max_tokens")),
            FakeStream(_final([_text_block(payload)])),
        ]
    )
    with _fake_client(fake):
        result = structured_call("claude-sonnet-5", "sys", "user", Sample, 16000)

    assert result == Sample(name="Jane", count=3)
    assert fake.stream_count == 3  # native 400 + text-json + one retry
    # text-json starts at DIRECT_TEXT_JSON_START_TOKENS (not the small native
    # generate budget) since Sonnet auto-thinks on this path too; the retry
    # bumps from there.
    assert fake.requests[1]["max_tokens"] == config.DIRECT_TEXT_JSON_START_TOKENS
    assert fake.requests[2]["max_tokens"] == min(
        config.DIRECT_TEXT_JSON_START_TOKENS * 4, config.GATEWAY_MAX_TOKENS
    )


def test_text_json_thinking_exhaustion_retries_only_once(monkeypatch):
    """The retry is bounded: a second thinking-exhaustion fails loudly instead
    of recursing forever at the bumped ceiling."""
    monkeypatch.setattr(config, "API_BASE_URL", None)  # direct Anthropic API
    monkeypatch.setattr("tools.llm.time.sleep", lambda _s: None)
    fake = FakeMessages(
        [
            _bad_request("Schema is too complex."),
            FakeStream(_final([_thinking_block()], stop_reason="max_tokens")),
            FakeStream(_final([_thinking_block()], stop_reason="max_tokens")),
        ]
    )
    with _fake_client(fake):
        try:
            structured_call("claude-sonnet-5", "sys", "user", Sample, 16000)
        except RuntimeError as exc:
            assert "no valid JSON" in str(exc)
        else:
            raise AssertionError("expected RuntimeError after the single retry")
    assert fake.stream_count == 3  # native 400 + text-json + one retry, no more
    assert fake.requests[1]["max_tokens"] == config.DIRECT_TEXT_JSON_START_TOKENS
    assert fake.requests[2]["max_tokens"] == min(
        config.DIRECT_TEXT_JSON_START_TOKENS * 4, config.GATEWAY_MAX_TOKENS
    )
