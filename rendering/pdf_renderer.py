"""PDF renderer — local, deterministic, no external service.

Replaces the n8n pipeline's external https://api.pdfshift.io dependency with
Playwright headless Chromium (Letter format, 18mm top/bottom, 14mm left/right).

Browser resolution order (robust to a blocked Playwright CDN, which happened
on this machine):
  1. Playwright's bundled Chromium.
  2. System Microsoft Edge (always present on Windows).
  3. System Google Chrome.
The chosen channel is cached per-process so a session doesn't re-probe.
"""
from __future__ import annotations

import io

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

import config

_CHANNEL_CACHE: str | None = None
_BUNDLED_ATTEMPTED = False


def _resolve_channel(playwright) -> str | None:
    """Return the first working channel: None==bundled, 'msedge', 'chrome'."""
    global _CHANNEL_CACHE, _BUNDLED_ATTEMPTED
    if _CHANNEL_CACHE is not None:
        return _CHANNEL_CACHE

    candidates: list[str | None] = [None]
    if not _BUNDLED_ATTEMPTED:
        candidates = [None, "msedge", "chrome"]
        _BUNDLED_ATTEMPTED = True
    else:
        candidates = ["msedge", "chrome"]

    for channel in candidates:
        try:
            args: dict = {}
            if channel is not None:
                args["channel"] = channel
            browser = playwright.chromium.launch(**args)
            browser.close()
            _CHANNEL_CACHE = channel
            return channel
        except PlaywrightError:
            continue
    _CHANNEL_CACHE = "__none__"
    return None


def render_pdf(html_doc: str) -> bytes:
    """Render `html_doc` (a full HTML document) to a Letter-format PDF byte
    string with the configured margins."""
    with sync_playwright() as p:
        channel = _resolve_channel(p)
        launch_args: dict = {}
        if channel is not None:
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
