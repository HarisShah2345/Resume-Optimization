"""Unit tests for the PDF renderer's browser-resolution logic.

`render_pdf` itself needs a real Playwright install, so we mock the
`sync_playwright` context manager and drive `_resolve_channel` /
`render_pdf` against a fake Chromium launcher. These cover the
channel-selection cache and the no-browser failure path without a browser.
"""
from __future__ import annotations

import io
import types
from unittest import mock

import pytest

import rendering.pdf_renderer as pdf_renderer

_BUNDLED = pdf_renderer._BUNDLED
_NO_BROWSER = pdf_renderer._NO_BROWSER


class _FakeBrowser:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def new_page(self):
        return _FakePage()

    def pdf(self, **kwargs):
        return b"%PDF-1.4 fake"


class _FakePage:
    def set_content(self, *a, **k):
        pass

    def pdf(self, **kwargs):
        return b"%PDF-1.4 fake"


class _FakeChromium:
    """Records every `launch` call; optionally raises on selected channels."""

    def __init__(self, fail_channels=()):
        # fail_channels: iterable of channel values that should raise PlaywrightError.
        self.fail_channels = set(fail_channels)
        self.launches: list[dict] = []

    def launch(self, **kwargs):
        self.launches.append(kwargs)
        # Bundled Chromium launches with no `channel` kwarg (empty kwargs).
        channel = kwargs.get("channel", _BUNDLED)
        if channel in self.fail_channels:
            raise pdf_renderer.PlaywrightError("boom")
        return _FakeBrowser()


class _FakePlaywright:
    def __init__(self, chromium):
        self.chromium = chromium

    def stop(self):
        pass


def _reset_module_state():
    """Restore module-level cache between tests so they're independent."""
    pdf_renderer._CHANNEL_CACHE = None
    pdf_renderer._BUNDLED_ATTEMPTED = False
    pdf_renderer._WEASY_IMPORT_TRIED = False
    pdf_renderer._WEASYPRINT_CACHE = None


def _install_fake_playwright(fake_chromium, monkeypatch):
    pw = _FakePlaywright(fake_chromium)
    # sync_playwright() is a context manager returning `pw`.
    monkeypatch.setattr(
        pdf_renderer,
        "sync_playwright",
        lambda: _CmrCtx(pw),
    )


class _CmrCtx:
    """Minimal context-manager wrapper returning the fake Playwright object."""

    def __init__(self, pw):
        self._pw = pw

    def __enter__(self):
        return self._pw

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _reset():
    _reset_module_state()
    yield
    _reset_module_state()


def test_resolve_channel_caches_first_working(fake_chromium, monkeypatch):
    _install_fake_playwright(fake_chromium, monkeypatch)
    pw = _FakePlaywright(fake_chromium)
    assert pdf_renderer._resolve_channel(pw) == _BUNDLED  # bundled works
    # Second call must hit the cache: no new launch attempts.
    assert pdf_renderer._resolve_channel(pw) == _BUNDLED
    assert len(fake_chromium.launches) == 1  # exactly one launch across both calls


def test_resolve_channel_falls_back_to_msedge(monkeypatch):
    # Bundled launch raises, msedge works.
    fake = _FakeChromium(fail_channels={_BUNDLED})
    _install_fake_playwright(fake, monkeypatch)
    pw = _FakePlaywright(fake)
    assert pdf_renderer._resolve_channel(pw) == "msedge"
    # Bundled probes with no `channel` kwarg; msedge probes with channel="msedge".
    channels = [l.get("channel", _BUNDLED) for l in fake.launches]
    assert channels == [_BUNDLED, "msedge"]


def test_resolve_channel_all_fail_raises_runtimeerror(monkeypatch):
    fake = _FakeChromium(fail_channels={_BUNDLED, "msedge", "chrome"})
    _install_fake_playwright(fake, monkeypatch)
    pw = _FakePlaywright(fake)
    with pytest.raises(RuntimeError, match="No Chromium-based browser available"):
        pdf_renderer._resolve_channel(pw)
    # Cache must record the failure so a second call raises the same, not retry.
    with pytest.raises(RuntimeError, match="No Chromium-based browser available"):
        pdf_renderer._resolve_channel(pw)
    # No-browser sentinel never leaks into a launch call.
    assert _NO_BROWSER not in [l.get("channel", _BUNDLED) for l in fake.launches]


def test_render_pdf_uses_bundled_when_no_channel_kwarg(monkeypatch):
    fake = _FakeChromium()
    _install_fake_playwright(fake, monkeypatch)
    out = pdf_renderer.render_pdf("<html></html>")
    assert out == b"%PDF-1.4 fake"
    # Bundled means: launch() called with no `channel` kwarg.
    assert fake.launches[-1] == {}


def test_render_pdf_no_browser_raises(monkeypatch):
    fake = _FakeChromium(fail_channels={_BUNDLED, "msedge", "chrome"})
    _install_fake_playwright(fake, monkeypatch)
    with pytest.raises(RuntimeError, match="No Chromium-based browser available"):
        pdf_renderer.render_pdf("<html></html>")


# ---------------------------------------------------------------------------
# weasyprint backend (import-time optional, mocked at the import boundary)
# ---------------------------------------------------------------------------
class _FakeWeasyHTML:
    """Mimics weasyprint.HTML(string=...): write_pdf() returns canned bytes."""

    def __init__(self, string):
        self.string = string
        _FakeWeasyHTML.last_string = string

    def write_pdf(self):
        return _FakeWeasyHTML.pdf_bytes

    pdf_bytes: bytes = b"%PDF-1.7 weasyprint-fake"


class _FakeWeasyModule:
    """Stand-in for the `weasyprint` module returned by _import_weasyprint."""

    HTML = _FakeWeasyHTML


def _stub_weasyprint(monkeypatch, available=True, html_cls=None, write_pdf=None):
    """Patch `_import_weasyprint` to return a fake (or to fail)."""
    if available:
        if html_cls is None:
            html_cls = _FakeWeasyHTML
            if write_pdf is not None:
                _FakeWeasyHTML.pdf_bytes = write_pdf  # type: ignore[attr-defined]
        fake_mod = types.SimpleNamespace(HTML=html_cls)
        monkeypatch.setattr(pdf_renderer, "_import_weasyprint", lambda: fake_mod)
    else:
        monkeypatch.setattr(
            pdf_renderer,
            "_import_weasyprint",
            lambda: (_ for _ in ()).throw(ImportError("no weasyprint")),
        )


def test_render_pdf_weasyprint_missing_returns_runtime_error():
    """If weasyprint isn't importable, the backend reports failure via a
    returned RuntimeError (never raised) so callers can chain fallbacks.

    weasyprint genuinely isn't installed in this env, so _import_weasyprint
    returns None after a real (one-shot) import attempt.
    """
    err = pdf_renderer.render_pdf_weasyprint("<html></html>")
    assert isinstance(err, RuntimeError)
    assert "weasyprint is not available" in str(err)


def test_render_pdf_weasyprint_success_returns_bytes(monkeypatch):
    _stub_weasyprint(monkeypatch, html_cls=_FakeWeasyHTML)
    out = pdf_renderer.render_pdf_weasyprint("<html><head></head><body>hi</body></html>")
    assert isinstance(out, bytes)
    assert out == b"%PDF-1.7 weasyprint-fake"
    # The @page margin rule was injected into the document.
    assert "@page" in _FakeWeasyHTML.last_string  # type: ignore[attr-defined]


def test_render_pdf_weasyprint_write_failure_returns_runtime_error(monkeypatch):
    """A weasyprint write_pdf() exception is wrapped, not raised."""

    class _BoomHTML:
        def __init__(self, string):
            pass

        def write_pdf(self):
            raise ValueError("cairo not found")

    _stub_weasyprint(monkeypatch, html_cls=_BoomHTML)
    err = pdf_renderer.render_pdf_weasyprint("<html></html>")
    assert isinstance(err, RuntimeError)
    assert "cairo not found" in str(err) or "weasyprint backend failed" in str(err)


def test_render_pdf_playwright_failure_falls_to_weasyprint(monkeypatch):
    """Playwright unavailable but weasyprint works → we get weasyprint bytes,
    no exception escapes (proves the backend chain)."""
    fake_pw_chromium = _FakeChromium(fail_channels={_BUNDLED, "msedge", "chrome"})
    _install_fake_playwright(fake_pw_chromium, monkeypatch)
    _stub_weasyprint(monkeypatch, html_cls=_FakeWeasyHTML)
    out = pdf_renderer.render_pdf("<html></html>")
    assert out == b"%PDF-1.7 weasyprint-fake"


def test_render_pdf_all_backends_fail_raises(monkeypatch):
    """Playwright fails AND weasyprint fails → RuntimeError raised (the
    contract render_node relies on for its HTML-only graceful path)."""
    fake_pw_chromium = _FakeChromium(fail_channels={_BUNDLED, "msedge", "chrome"})
    _install_fake_playwright(fake_pw_chromium, monkeypatch)

    class _BoomHTML2:
        def __init__(self, string):
            pass

        def write_pdf(self):
            raise OSError("no system libs")

    _stub_weasyprint(monkeypatch, html_cls=_BoomHTML2)
    with pytest.raises(RuntimeError) as exc_info:
        pdf_renderer.render_pdf("<html></html>")
    # The re-raised error is the original Playwright RuntimeError — its
    # message names the missing browser, which is the user-facing cue.
    assert "No Chromium-based browser available" in str(exc_info.value)


@pytest.fixture
def fake_chromium():
    _reset_module_state()
    return _FakeChromium()
