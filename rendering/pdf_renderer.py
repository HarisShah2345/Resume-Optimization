"""PDF renderer — local, deterministic, no external service.

Replaces the n8n pipeline's external https://api.pdfshift.io dependency. PDF
backend order (robust to environments that lack a browser, e.g. Streamlit
Community Cloud which can't install Playwright browsers):

  1. Playwright headless Chromium (Letter format, 18mm top/bottom, 14mm
     left/right). Resolution order: bundled Chromium → system Edge → system
     Chrome. The chosen channel is cached per-process so a session doesn't
     re-probe.
  2. weasyprint — a pure-Python (pydyf-backed) HTML→PDF fallback so a browser
     isn't strictly required. Import is optional: if the package (or its
     system libs) isn't present, this backend is skipped.
  3. HTML-only — when no backend can produce a PDF, `render_pdf` raises
     RuntimeError so the caller (graph/nodes.render_node) can fall back to
     delivering the complete HTML to the UI instead of failing the run.

"""
from __future__ import annotations

import io

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

import config

# Cache sentinels for `_resolve_channel`. None means "unprobed"; using distinct
# string sentinels (not None) lets the cache hit the bundled case, which the
# original `if _CHANNEL_CACHE is not None` check silently skipped.
_BUNDLED: str = "__bundled__"
_NO_BROWSER: str = "__none__"
_CHANNEL_CACHE: str | None = None
_BUNDLED_ATTEMPTED = False

# weasyprint import is memoized once (None means "not yet tried"). It shells
# out to Pango (needs libpango/libpangocairo/libcairo2/libgdk-pixbuf2.0 via
# packages.txt at the repo root on Streamlit Cloud), so we never want to
# retry a failed import every call.
_WEASYPRINT_CACHE: object = None  # module | None | False
_WEASY_IMPORT_TRIED: bool = False


def _resolve_channel(playwright) -> str:
    """Return the first working channel: `_BUNDLED`==bundled, 'msedge', 'chrome'.

    Raises RuntimeError if no Chromium-based browser is launchable — including
    a prior total failure cached as `_NO_BROWSER`, so the caller never hands
    the sentinel to `launch(channel=...)` (which Playwright does not understand).
    """
    global _CHANNEL_CACHE, _BUNDLED_ATTEMPTED
    if _CHANNEL_CACHE == _NO_BROWSER:
        raise RuntimeError(
            "No Chromium-based browser available for PDF rendering. Install one "
            "via `uv run playwright install chromium` (bundled), or install "
            "Microsoft Edge / Google Chrome on this machine."
        )
    if _CHANNEL_CACHE is not None:
        return _CHANNEL_CACHE

    candidates: list[str] = [_BUNDLED]
    if not _BUNDLED_ATTEMPTED:
        candidates = [_BUNDLED, "msedge", "chrome"]
        _BUNDLED_ATTEMPTED = True
    else:
        candidates = ["msedge", "chrome"]

    for channel in candidates:
        try:
            # Bundled Chromium launches with no `channel` kwarg; named channels
            # pass it explicitly so Playwright picks msedge/chrome.
            args: dict = {}
            if channel != _BUNDLED:
                args["channel"] = channel
            browser = playwright.chromium.launch(**args)
            browser.close()
            _CHANNEL_CACHE = channel
            return channel
        except PlaywrightError:
            continue
    _CHANNEL_CACHE = _NO_BROWSER
    raise RuntimeError(
        "No Chromium-based browser available for PDF rendering. Install one "
        "via `uv run playwright install chromium` (bundled), or install "
        "Microsoft Edge / Google Chrome on this machine."
    )


def _import_weasyprint():
    """Return the weasyprint module, or None if it can't be imported.

    Memoized per process: weasyprint is a real system dependency on
    cairocffi-based versions (needs libpango/libcairo), so a failed import
    must not be retried every call — the backend is simply skipped.
    """
    global _WEASYPRINT_CACHE, _WEASY_IMPORT_TRIED
    if _WEASY_IMPORT_TRIED:
        return None if _WEASYPRINT_CACHE is False else _WEASYPRINT_CACHE
    _WEASY_IMPORT_TRIED = True
    try:
        # Imported lazily + memoized so the module isn't a hard requirement.
        import weasyprint  # type: ignore
    except Exception:
        _WEASYPRINT_CACHE = False
        return None
    _WEASYPRINT_CACHE = weasyprint
    return weasyprint


def _page_margin_css() -> str:
    """Build the `@page { margin: ... }` CSS mirroring Playwright's mm margins
    so the weasyprint fallback produces Letter output at the same geometry.
    """
    return (
        f"@page {{ "
        f"size: {config.PAGE_FORMAT}; "
        f"margin: {config.PDF_MARGIN_TOP_MM}mm "
        f"{config.PDF_MARGIN_RIGHT_MM}mm "
        f"{config.PDF_MARGIN_BOTTOM_MM}mm "
        f"{config.PDF_MARGIN_LEFT_MM}mm; "
        f"}}"
    )


def render_pdf_weasyprint(html_doc: str):
    """Render `html_doc` to a Letter-format PDF via weasyprint.

    Returns `bytes` on success, or a `RuntimeError` INSTANCE (not raised) on
    failure — callers chain backends and only surface a raised RuntimeError as
    the final "no PDF anywhere" signal.
    """
    weasyprint = _import_weasyprint()
    if weasyprint is None:
        return RuntimeError(
            "weasyprint is not available; cannot render PDF via this backend."
        )
    try:
        # write_pdf() takes the full HTML doc string; weasyprint inlines the
        # @page margin rule we append so geometry matches Playwright.
        styled = (
            html_doc.replace(
                "<head>",
                f"<head><style>{_page_margin_css()}</style>",
                1,
            )
            if "<head>" in html_doc
            else f'<style>{_page_margin_css()}</style>{html_doc}'
        )
        out = weasyprint.HTML(string=styled).write_pdf()
        return out
    except Exception as exc:  # noqa: BLE001 — weasyprint raises varied errors
        return RuntimeError(
            f"weasyprint backend failed: {exc} (install libpango/libcairo2 if "
            f"missing)."
        )


def render_pdf(html_doc: str) -> bytes:
    """Render `html_doc` (a full HTML document) to a Letter-format PDF byte
    string.

    Backend chain:
      1. Playwright headless Chromium (bundled → Edge → Chrome).
      2. weasyprint (pure-Python, no browser needed).
      3. If both fail, raise RuntimeError with a descriptive message — the
         caller (graph/nodes.render_node) catches it and delivers the
         complete HTML to the UI as a graceful fallback.

    `_resolve_channel` raises RuntimeError if no browser is available, so by the
    time we reach `launch` the channel is always `_BUNDLED` or a real channel
    name ('msedge' / 'chrome'). `_BUNDLED` maps to launching with no `channel`
    kwarg — i.e. Playwright's default bundled Chromium.
    """
    try:
        with sync_playwright() as p:
            channel = _resolve_channel(p)
            launch_args: dict = {}
            if channel != _BUNDLED:
                launch_args["channel"] = channel
            browser = p.chromium.launch(**launch_args)
            try:
                page = browser.new_page()
                page.set_content(html_doc, wait_until="load")
                pdf_bytes: bytes = page.pdf(
                    format=config.PAGE_FORMAT,
                    margin={
                        "top": f"{config.PDF_MARGIN_TOP_MM}mm",
                        "bottom": f"{config.PDF_MARGIN_BOTTOM_MM}mm",
                        "left": f"{config.PDF_MARGIN_LEFT_MM}mm",
                        "right": f"{config.PDF_MARGIN_RIGHT_MM}mm",
                    },
                    print_background=True,
                )
                return io.BytesIO(pdf_bytes).getvalue()
            finally:
                browser.close()
    except RuntimeError as browser_err:
        # Playwright (or its channel probing) is unavailable — try weasyprint.
        weasy_result = render_pdf_weasyprint(html_doc)
        if isinstance(weasy_result, bytes):
            return weasy_result
        # Both backends failed: re-raise the original Playwright RuntimeError so
        # the message stays honest about what's missing (a browser + weasyprint).
        raise browser_err
