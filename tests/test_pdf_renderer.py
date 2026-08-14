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


@pytest.fixture
def fake_chromium():
    _reset_module_state()
    return _FakeChromium()
