"""Thin wrapper over the official `anthropic` SDK (never an OpenAI-compatible
shim).

One uniform entry point for every LLM stage, with two structured-output
mechanisms:

1. NATIVE (direct Anthropic API) — `messages.stream()` + `output_format`
   (Pydantic) → validated instance via `get_final_message().parsed_output`.
2. GATEWAY (any non-default `ANTHROPIC_BASE_URL`) — forced tool-use
   constrained decoding, the universally-supported technique. Gateways strip
   Anthropic's `output_format` feature, so we ask the model to call a
   `structured_output` tool whose input schema is the Pydantic schema, then
   validate the tool_use block's input dict. Verified against the local combo
   gateway on this machine (`free-bundle` route).
3. TEXT-JSON (last resort) — when a schema is rejected (`400 Schema is too
   complex.` or `400 Grammar compilation timed out` — both grammar-based limits
   that the full ResumeContent schema triggers), the call is re-issued with NO
   tools: the model emits the JSON as ordinary text and we extract + validate
   it. The direct API's native `output_format` rejects big schemas too, and its
   tool-use path shares the same grammar machinery, so the native path skips
   tool-use and falls straight to text-JSON. Keeps large schemas (e.g.
   ResumeContent) working on either path. Once a schema is rejected on the
   native path, that outcome is memoized for the process, so later calls for the
   same schema (e.g. every repair/validate pass in one agent run) skip the
   doomed ~3-minute grammar-compile rejection and go straight to text-JSON.

Both paths stream, so the 10-minute max_tokens guard never trips and `on_token`
feeds live deltas to the UI (visible text on the native path; tool-arg JSON on
a gateway). Transient 5xx / pool-exhaustion / connection errors from gateways
are retried with backoff, and streaming is suppressed until the final attempt
so a failed attempt never leaks partial deltas into the UI.

API-drift notes respected:
  - `budget_tokens` is removed on current models — we use `thinking={"type":"adaptive"}`.
  - `temperature`/`top_p`/`top_k` are removed on Sonnet 5 — we never pass them.
  - Prompt-cache `cache_control` lives INSIDE the system content block (spec),
    never as a top-level request field.
"""
from __future__ import annotations

import concurrent.futures
import json
import time
from typing import Callable, Type, TypeVar

import anthropic
import httpx
from pydantic import BaseModel, ValidationError

import config

T = TypeVar("T", bound=BaseModel)

TokenCallback = Callable[[str], None]

_TOOL_NAME = "structured_output"

# Transient failures worth a retry with backoff — gateways sit behind pool
# schedulers that throw 5xx / connection drops when a combo route is exhausted.
# httpx.TransportError (ConnectError/ReadError/RemoteProtocolError, wrapping OS
# errors like "[Errno 104] Connection reset by peer") is included separately
# from anthropic.APIConnectionError: the SDK only wraps errors raised while
# *establishing* a request, not ones raised mid-stream while iterating SSE
# chunks (`Stream.__stream__` has no try/except around that loop) — verified
# live on Streamlit Cloud, whose egress drops connections mid-response.
_RETRYABLE = (
    anthropic.InternalServerError,
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    httpx.TransportError,
)
_MAX_ATTEMPTS = 3
# Wall-clock deadline per attempt. A gateway stream can stall indefinitely
# (connection ESTABLISHED, no delta ever arriving — verified live on
# free-bundle) and the SDK's read timeout is idle-based, so without a hard
# deadline one hung request would block the whole agent forever.
_ATTEMPT_TIMEOUT_SECONDS: int = 600

# Schemas already rejected by the direct API's native `output_format` grammar
# (`400 Schema is too complex.` / `400 Grammar compilation timed out`). Rejection
# is a property of the schema, not the moment, so once a schema 400s, every
# later call for it in this process skips the doomed native attempt. Keyed by
# `id(output_model)`; see `_native_call`.
_native_schema_failures: set[int] = set()

# The in-memory cache above resets every process — a fresh Streamlit rerun or
# `e2e_live.py` invocation re-pays the same doomed grammar-compile rejection
# (observed live: 3 to 26 minutes for the SAME schema, ResumeContent, since the
# server-side compile time is itself highly variable). Persist known-rejected
# schema names to disk so the cost is paid once ever per environment, not once
# per process. Keyed by class name (stable across processes; `id()` is not).
_SCHEMA_FAILURES_FILE = "native_schema_failures.json"


def _load_persisted_schema_failures() -> set[str]:
    try:
        path = config.TEMP_DIR / _SCHEMA_FAILURES_FILE
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


def _persist_schema_failure(name: str) -> None:
    try:
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        path = config.TEMP_DIR / _SCHEMA_FAILURES_FILE
        names = _load_persisted_schema_failures()
        names.add(name)
        path.write_text(json.dumps(sorted(names)), encoding="utf-8")
    except OSError:
        pass


_persisted_schema_failures: set[str] = _load_persisted_schema_failures()


def _client() -> anthropic.Anthropic:
    kwargs: dict = {}
    if config.API_BASE_URL:
        kwargs["base_url"] = config.API_BASE_URL
    return anthropic.Anthropic(api_key=config.require_api_key(), **kwargs)


def _log(msg: str) -> None:
    """Append one line to the LLM request log (~/.resume-optimizer/llm.log).

    Observability only — never raises, never blocks a request. Lets a running
    app (Streamlit) be diagnosed from outside by tailing the file.
    """
    try:
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.TEMP_DIR / "llm.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except OSError:
        pass


def _system_arg(system: str) -> list[dict]:
    """System prompt as content blocks with prompt-cache control (spec-correct:
    `cache_control` is a property of a content block, never a top-level field)."""
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _drain_stream(stream, on_token) -> None:
    """Forward every visible delta to on_token (no-op when on_token is None).

    On a gateway the structured output arrives as tool-arg JSON streamed via
    `input_json_delta` (the SDK exposes it as `delta.partial_json`); on the
    native path it is `delta.text`. Both are forwarded so the UI streams live.
    """
    if on_token is None:
        return
    for event in stream:
        if event.type != "content_block_delta":
            continue
        delta = event.delta
        chunk = getattr(delta, "partial_json", None)
        if chunk is None:
            chunk = getattr(delta, "text", None)
        if chunk:
            on_token(chunk)


def _call_with_retry(fn: Callable[[TokenCallback | None], anthropic.types.Message], on_token):
    """Run `fn(forward)`, retrying transient failures with backoff.

    Deltas streamed by an attempt are buffered and only flushed to `on_token`
    if that attempt SUCCEEDS — so the UI streams live in the happy path (first
    attempt succeeds) but a failed mid-stream attempt never leaks partial
    deltas into it.

    Each attempt also runs under a wall-clock deadline (`_ATTEMPT_TIMEOUT_SECONDS`).
    The SDK's own read timeout is idle-based — a stalled gateway stream that
    keeps the connection open but sends nothing never trips it — so without a
    deadline a hung request would block the agent forever. On timeout the
    attempt's worker thread is abandoned (`shutdown(wait=False)`); it unblocks
    whenever the gateway finally responds or the connection drops, and its
    buffered deltas are discarded with the attempt.
    """
    last: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        buf: list[str] = []

        def _forward(chunk: str) -> None:
            buf.append(chunk)

        outcome: str = "ok"  # "ok" | "retryable" | "timeout"
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(fn, _forward)
            try:
                result = future.result(timeout=_ATTEMPT_TIMEOUT_SECONDS)
            except TimeoutError:
                outcome = "timeout"
                last = TimeoutError(
                    f"LLM call exceeded the {_ATTEMPT_TIMEOUT_SECONDS}s wall-clock "
                    f"deadline (attempt {attempt + 1}/{_MAX_ATTEMPTS})."
                )
                _log(
                    f"retry timeout after {_ATTEMPT_TIMEOUT_SECONDS}s "
                    f"(attempt {attempt + 1}/{_MAX_ATTEMPTS}) — abandoning hung stream"
                )
        except _RETRYABLE as exc:
            last = exc
            outcome = "retryable"
            _log(
                f"retry {exc.__class__.__name__}: {str(exc)[:120]} "
                f"(attempt {attempt + 1}/{_MAX_ATTEMPTS})"
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if outcome == "ok":
            if on_token is not None:
                for chunk in buf:
                    on_token(chunk)
            return result
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(2 * (attempt + 1))
        # else: last attempt — fall through to the raise after the loop
    assert last is not None
    raise last


# ---------------------------------------------------------------------------
# Gateway path: forced tool-use constrained decoding
# ---------------------------------------------------------------------------
def _tool_use_request(model, system, user, output_model, max_tokens, thinking) -> dict:
    request: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": _system_arg(system),
        "messages": [{"role": "user", "content": user}],
        "tools": [
            {
                "name": _TOOL_NAME,
                "description": (
                    "Emit exactly one JSON object matching the required schema. "
                    "Do not emit any prose around it."
                ),
                "input_schema": output_model.model_json_schema(),
            }
        ],
        "tool_choice": {"type": "tool", "name": _TOOL_NAME},
    }
    if thinking:
        request["thinking"] = {"type": "adaptive"}
    return request


def _decode_nested_json(value: object) -> object:
    """Recursively decode JSON-encoded strings back into structures.

    Some gateway backends double-encode nested structured fields — verified
    live on free-bundle: the `contact` object of ResumeContent arrived as a
    JSON string (`"contact": "{\"name\": ...}"`) instead of a nested dict,
    which made `model_validate` reject the whole document. Walk any tool_use
    input (or extracted JSON text) and decode every string that is itself a
    JSON object/array before validating, so the shapes Pydantic expects are
    actually dicts/lists at every depth.

    Only `{`/`[`-prefixed strings are touched — scalar-looking strings
    (names, dates, counts) are left exactly as-is.
    """
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return _decode_nested_json(json.loads(s))
            except ValueError:
                return value
        return value
    if isinstance(value, dict):
        return {k: _decode_nested_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_nested_json(v) for v in value]
    return value


def _extract_json_object(text: str) -> str | None:
    """Return the outermost {...} span in text (best-effort JSON recovery)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def _is_schema_rejection(exc: anthropic.BadRequestError) -> bool:
    """True when a 400 means the OUTPUT SCHEMA — not the request — is the problem.

    The grammar-based structured-output features reject large schemas in two
    ways, both verified live against the full ResumeContent schema:
      - `Schema is too complex.`          (fails fast)
      - `Grammar compilation timed out`   (the server spends minutes compiling
                                          the grammar, then 400s)
    Tool-use `input_schema` uses the same grammar machinery, so neither is a
    transient failure that retry can fix — the schema-less text-JSON path is the
    recovery. Genuine request errors (bad model name, bad key) still propagate.
    """
    msg = str(exc).lower()
    return "too complex" in msg or "grammar compilation" in msg


def _schema_skeleton(output_model: Type[T]) -> str:
    """A compact, field-typed skeleton of `output_model` for the text-JSON prompt.

    In free-text mode the model cannot see the schema — `output_format`/tool-use
    are what normally constrain it — so it GUESSES the shape. Verified live on
    the direct API: ResumeContent came back with `skills` as a flat string list
    instead of `[{category, items}]`, which then failed Pydantic validation.
    Rendering every field with its resolved type keeps the model on-shape.
    """
    schema = output_model.model_json_schema()
    defs = schema.get("$defs", {})

    def _resolve(ref: str) -> dict:
        return defs.get(ref.rsplit("/", 1)[-1], {})

    def _render(fs: dict, seen: tuple = ()) -> str:
        t = fs.get("type")
        if t == "array":
            return f"array of {_render(fs.get('items', {}), seen)}"
        if t == "object":
            props = fs.get("properties", {})
            inner = ", ".join(f"{k}: {_render(v, seen)}" for k, v in props.items())
            return "{" + inner + "}"
        ref = fs.get("$ref")
        if ref:
            name = ref.rsplit("/", 1)[-1]
            if name in seen:
                return name  # cycle guard
            return _render(_resolve(ref), seen + (name,))
        any_of = fs.get("anyOf")
        if any_of:
            return " | ".join(_render(x, seen) for x in any_of)
        return t or "any"

    props = schema.get("properties", {})
    if not props:
        return _render(schema)
    return "\n".join(f"- {name}: {_render(prop)}" for name, prop in props.items())


def _text_json_call(
    client,
    model,
    system,
    user,
    output_model,
    max_tokens,
    on_token,
    thinking: bool = False,
    _recovered: bool = False,
):
    """Plain-text JSON emission fallback (no tools).

    Used when the gateway rejects the forced tool-use `input_schema` as too
    complex (verified live on free-bundle: `400 invalid_request_error: Schema
    is too complex.` — the real Anthropic API limit on schema depth/size, and
    the full ResumeContent schema triggers it). With no `tools` in the request
    there is no schema-complexity check at all: the model emits the JSON as
    ordinary text, we extract the outermost {...} span and validate it.

    Omitting `thinking` is NOT the same as disabling it — verified live,
    Sonnet auto-thinks on this tool-less path anyway even when it's never
    requested, and `adaptive` thinking alone has no depth control, so a hard
    task (e.g. trimming a resume to satisfy the keyword-cap repair
    instruction) can run unbounded for 100-345s. But fully disabling thinking
    made that same repair fail to converge (verified live: 3/3 runs).
    `thinking.type.enabled` + `budget_tokens` is rejected outright on
    claude-sonnet-5 (400: "not supported for this model. Use
    thinking.type.adaptive and output_config.effort" — verified live), so the
    real bounded knob is `output_config.effort`: `adaptive` thinking capped at
    a low effort level — bounded reasoning instead of none or unbounded.
    Callers that don't need it (the cheap Haiku validate pass) leave
    `thinking=False` and get it disabled outright.
    """
    _log(f"text_json start model={model} max_tokens={max_tokens} thinking={thinking}")
    t0 = time.time()
    request: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": _system_arg(system),
        **(
            {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": config.DIRECT_TEXT_JSON_THINKING_EFFORT},
            }
            if thinking
            else {"thinking": {"type": "disabled"}}
        ),
        "messages": [
            {
                "role": "user",
                "content": (
                    user
                    + "\n\nRespond with ONLY a single valid JSON object matching "
                    "the required JSON shape below. No prose, no markdown fences, "
                    "no commentary — just the JSON.\n\n"
                    "## required JSON shape (field: type)\n"
                    + _schema_skeleton(output_model)
                ),
            }
        ],
    }

    def _run(cb):
        with client.messages.stream(**request) as stream:
            if cb is not None:
                for text in stream.text_stream:
                    cb(text)
            return stream.get_final_message()

    final = _call_with_retry(_run, on_token)
    text = "".join(b.text for b in final.content if getattr(b, "type", None) == "text")
    json_txt = _extract_json_object(text)
    if json_txt:
        try:
            result = output_model.model_validate(_decode_nested_json(json_txt))
            _log(f"text_json ok in {time.time() - t0:.1f}s")
            return result
        except ValidationError:
            pass

    # The model auto-thought away the whole budget without emitting the JSON
    # (verified live: content=['thinking'], stop_reason=max_tokens on the repair
    # pass). Retry exactly ONCE with a bumped budget so thinking + the JSON fit;
    # `_recovered` bounds the recursion so a second exhaustion fails loudly.
    if final.stop_reason == "max_tokens" and not _recovered:
        bumped = min(max_tokens * 4, config.GATEWAY_MAX_TOKENS)
        _log(
            f"text_json budget exhausted after {time.time() - t0:.1f}s "
            f"(content types: {[b.type for b in final.content]}), retrying at bumped={bumped}"
        )
        return _text_json_call(
            client,
            model,
            system,
            user,
            output_model,
            bumped,
            on_token,
            thinking=thinking,
            _recovered=True,
        )

    raise RuntimeError(
        f"Text-JSON fallback for {model} returned no valid JSON "
        f"(content types: {[b.type for b in final.content]}, "
        f"stop_reason={final.stop_reason})."
    )


def _tool_use_call(
    client, model, system, user, output_model, max_tokens, on_token, thinking, _recovered: bool = False
):
    request = _tool_use_request(model, system, user, output_model, max_tokens, thinking)
    _log(f"tool_use start model={model} max_tokens={max_tokens} recovered={_recovered}")
    t0 = time.time()

    def _run(cb):
        with client.messages.stream(**request) as stream:
            _drain_stream(stream, cb)
            return stream.get_final_message()

    try:
        final = _call_with_retry(_run, on_token)
    except anthropic.BadRequestError as exc:
        # Gateway refused the tool-use schema — the grammar built from a large
        # input_schema exceeds the backend's limits (400 "Schema is too
        # complex." / "Grammar compilation timed out"). Not a transient error,
        # so retry won't help — switch to the schema-less text-JSON path
        # instead of failing the whole run.
        if _is_schema_rejection(exc):
            _log(f"tool_use -> text-json fallback (schema rejected: {str(exc)[:80]})")
            return _text_json_call(
                client, model, system, user, output_model, max_tokens, on_token
            )
        raise
    for block in final.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == _TOOL_NAME:
            # Gateways sometimes return tool arguments as a JSON STRING instead
            # of a parsed dict, and occasionally double-encode nested fields as
            # JSON strings (verified live on free-bundle: the `contact` object
            # arrived as a string). `_decode_nested_json` normalizes both shapes.
            if isinstance(block.input, str):
                _log(f"tool_use ok in {time.time() - t0:.1f}s (string-typed input)")
            else:
                _log(f"tool_use ok in {time.time() - t0:.1f}s")
            return output_model.model_validate(_decode_nested_json(block.input))

    # A few gateway backends answer with bare JSON in text instead of a
    # tool_use block. Recover it before giving up.
    json_txt = _extract_json_object(
        "".join(b.text for b in final.content if getattr(b, "type", None) == "text")
    )
    if json_txt:
        try:
            result = output_model.model_validate(_decode_nested_json(json_txt))
            _log(f"tool_use ok via text-JSON recovery in {time.time() - t0:.1f}s")
            return result
        except ValidationError:
            pass

    # Combo gateways often AUTO-think even when thinking isn't requested, and a
    # big generation task can burn the whole token budget on thinking without
    # ever emitting the tool_use block (verified live: content=[thinking],
    # stop_reason=max_tokens). Retry exactly ONCE with thinking off and a larger
    # budget. `_recovered` bounds the recursion to a single retry, so a response
    # that is thinking-only even at the GATEWAY_MAX_TOKENS cap still gets one
    # more chance.
    if final.content and all(
        getattr(b, "type", None) == "thinking" for b in final.content
    ):
        if not _recovered:
            bumped = min(max_tokens * 4, config.GATEWAY_MAX_TOKENS)
            _log(
                f"tool_use -> thinking-only after {time.time() - t0:.1f}s, "
                f"retrying at bumped={bumped}"
            )
            return _tool_use_call(
                client,
                model,
                system,
                user,
                output_model,
                bumped,
                on_token=None,  # the failed pass streamed only (non-forwarded) thinking
                thinking=False,
                _recovered=True,
            )
        _log(f"tool_use thinking-only even after recovery (bumped={max_tokens})")

    _log(
        f"tool_use failed after {time.time() - t0:.1f}s: no {_TOOL_NAME} tool_use block "
        f"(content types: {[b.type for b in final.content]}, stop_reason={final.stop_reason})"
    )
    raise RuntimeError(
        f"Structured output from {model} returned no {_TOOL_NAME} tool_use block "
        f"(content types: {[b.type for b in final.content]}, "
        f"stop_reason={final.stop_reason})."
    )


# ---------------------------------------------------------------------------
# Native path: direct Anthropic API with output_format (Pydantic)
# ---------------------------------------------------------------------------
def _native_call(client, model, system, user, output_model, max_tokens, on_token, thinking):
    _log(f"native start model={model} max_tokens={max_tokens} thinking={thinking}")
    if id(output_model) in _native_schema_failures or output_model.__name__ in _persisted_schema_failures:
        _log(f"native -> text-json fallback (schema {output_model.__name__} previously rejected)")
        return _text_json_call(
            client,
            model,
            system,
            user,
            output_model,
            max(max_tokens, config.DIRECT_TEXT_JSON_START_TOKENS),
            on_token,
            thinking=thinking,
        )
    t0 = time.time()
    request: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": _system_arg(system),
        "messages": [{"role": "user", "content": user}],
        "output_format": output_model,
    }
    if thinking:
        request["thinking"] = {"type": "adaptive"}

    def _run(cb):
        with client.messages.stream(**request) as stream:
            if cb is not None:
                for text in stream.text_stream:
                    cb(text)
            return stream.get_final_message()

    try:
        final = _call_with_retry(_run, on_token)
    except anthropic.BadRequestError as exc:
        # The direct API's native `output_format` enforces a schema-complexity
        # limit and rejects large schemas (verified live against ResumeContent:
        # `400 "Schema is too complex."` then `400 "Grammar compilation timed
        # out"`). Not a transient error — retry won't help, and the tool-use
        # input_schema path shares the same grammar machinery, so it would fail
        # the same way after the same slow compile. Recover DIRECTLY via the
        # schema-less text-JSON path. Nothing streamed before the 400, so
        # on_token is unconsumed and safe to pass through.
        if _is_schema_rejection(exc):
            # Memoize so every later call for this schema (repair passes,
            # validation passes) skips the doomed native attempt instead of
            # re-paying the slow grammar-compile rejection.
            _native_schema_failures.add(id(output_model))
            if output_model.__name__ not in _persisted_schema_failures:
                _persisted_schema_failures.add(output_model.__name__)
                _persist_schema_failure(output_model.__name__)
            _log(f"native -> text-json fallback (schema rejected: {str(exc)[:80]})")
            return _text_json_call(
                client,
                model,
                system,
                user,
                output_model,
                max(max_tokens, config.DIRECT_TEXT_JSON_START_TOKENS),
                on_token,
                thinking=thinking,
            )
        raise
    parsed = getattr(final, "parsed_output", None)
    if parsed is not None:
        _log(f"native ok in {time.time() - t0:.1f}s")
        return parsed

    # output_format was silently ignored (e.g. the base URL is a gateway after
    # all) and its text already streamed — recover via tool-use decoding, but
    # without double-streaming (the first attempt consumed on_token).
    _log(f"native -> tool_use fallback in {time.time() - t0:.1f}s (no parsed_output)")
    return _tool_use_call(client, model, system, user, output_model, max_tokens, None, thinking)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def structured_call(
    model: str,
    system: str,
    user: str,
    output_model: Type[T],
    max_tokens: int,
    *,
    on_token: TokenCallback | None = None,
    thinking: bool = False,
) -> T:
    """Run a structured-output call and return a validated `output_model`.

    `on_token` receives each streamed delta (the UI streams the generation
    stage live). `thinking` enables adaptive extended thinking for models that
    support it.
    """
    client = _client()
    if config.is_gateway():
        _log(f"structured_call gateway=True model={model} output={output_model.__name__}")
        return _tool_use_call(
            client, model, system, user, output_model, max_tokens, on_token, thinking
        )
    _log(f"structured_call gateway=False model={model} output={output_model.__name__}")
    return _native_call(
        client, model, system, user, output_model, max_tokens, on_token, thinking
    )


def simple_text(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    *,
    on_token: TokenCallback | None = None,
) -> str:
    """Free-text call (no schema) — used for small inline explanations."""
    client = _client()

    def _run(cb):
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=_system_arg(system),
            messages=[{"role": "user", "content": user}],
        ) as stream:
            if cb is not None:
                for text in stream.text_stream:
                    cb(text)
            return stream.get_final_message()

    final = _call_with_retry(_run, on_token)
    return "".join(b.text for b in final.content if b.type == "text")
