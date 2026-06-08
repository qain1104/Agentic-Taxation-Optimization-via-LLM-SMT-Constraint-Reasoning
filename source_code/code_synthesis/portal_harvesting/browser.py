from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from .models import FormControl

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
    _HAS_PLAYWRIGHT = True
except Exception:  # pragma: no cover
    PlaywrightTimeoutError = RuntimeError
    sync_playwright = None
    _HAS_PLAYWRIGHT = False

DEFAULT_TIMEOUT_MS = 30000
DEFAULT_WAIT_AFTER_LOAD_MS = 1200


def _html_to_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = soup.get_text("\n", strip=True)
    return text, title


def parse_form_controls(html: str) -> List[FormControl]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[FormControl] = []
    for tag_name in ("input", "select", "textarea"):
        for el in soup.find_all(tag_name):
            control_type = ""
            if tag_name == "input":
                control_type = (el.get("type") or "text").strip().lower()
            elif tag_name == "select":
                control_type = "select"
            else:
                control_type = "textarea"

            nearby = []
            parent = el.parent
            if parent:
                parent_text = parent.get_text(" ", strip=True)
                if parent_text:
                    nearby.append(parent_text[:200])
            out.append(
                FormControl(
                    tag=tag_name,
                    control_type=control_type,
                    name=(el.get("name") or "").strip(),
                    element_id=(el.get("id") or "").strip(),
                    placeholder=(el.get("placeholder") or "").strip(),
                    aria_label=(el.get("aria-label") or "").strip(),
                    title=(el.get("title") or "").strip(),
                    value=(el.get("value") or "").strip(),
                    text_nearby=" | ".join(nearby),
                )
            )
    return out


def load_html_snapshot(path: str) -> Tuple[str, str, str, List[FormControl], str]:
    p = Path(path).expanduser()
    html = p.read_text(encoding="utf-8")
    text, title = _html_to_text(html)
    return html, text, title, parse_form_controls(html), "snapshot"


def fetch_html_with_requests(url: str, timeout_s: int = 30) -> Tuple[str, str, str, List[FormControl], str]:
    resp = requests.get(
        url,
        timeout=timeout_s,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    resp.raise_for_status()
    html = resp.text
    text, title = _html_to_text(html)
    return html, text, title, parse_form_controls(html), "requests"


def fetch_html_with_playwright(url: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> Tuple[str, str, str, List[FormControl], str]:
    if not _HAS_PLAYWRIGHT:
        raise RuntimeError("Playwright is not available")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(
                viewport={"width": 1440, "height": 2200},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                locale="zh-TW",
            )
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except PlaywrightTimeoutError:
                pass
            page.wait_for_timeout(DEFAULT_WAIT_AFTER_LOAD_MS)
            html = page.content()
            text = page.locator("body").inner_text(timeout=5000)
            title = page.title()
            return html, text, title, parse_form_controls(html), "playwright"
        finally:
            browser.close()


def fetch_portal_page(url: str, html_snapshot: Optional[str] = None, timeout_ms: int = DEFAULT_TIMEOUT_MS):
    if html_snapshot:
        return load_html_snapshot(html_snapshot)
    errors = []
    if _HAS_PLAYWRIGHT:
        try:
            return fetch_html_with_playwright(url, timeout_ms=timeout_ms)
        except Exception as e:  # pragma: no cover
            errors.append(f"playwright: {type(e).__name__}: {e}")
    try:
        return fetch_html_with_requests(url)
    except Exception as e:
        errors.append(f"requests: {type(e).__name__}: {e}")
    raise RuntimeError("Failed to fetch portal page: " + " | ".join(errors))
