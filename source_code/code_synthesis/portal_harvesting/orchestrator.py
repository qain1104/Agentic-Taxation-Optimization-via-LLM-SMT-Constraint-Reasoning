from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .browser import fetch_portal_page
from .extractors import harvest_with_spec
from .models import RenderedPortalArtifacts
from .renderers import render_problem_text, render_refs_text
from .specs import get_schema_spec


def harvest_and_render_portal(
    *,
    schema: str,
    url: Optional[str] = None,
    html_path: Optional[str] = None,
    timeout_ms: int = 30000,
) -> RenderedPortalArtifacts:
    spec = get_schema_spec(schema)
    source_url = url or spec.get("source_url") or ""
    if not source_url and not html_path:
        raise ValueError(f"schema={schema} requires either a url or html_path")

    html, text, page_title, controls, fetch_mode = fetch_portal_page(
        source_url,
        html_snapshot=html_path,
        timeout_ms=timeout_ms,
    )
    _ = html

    harvested = harvest_with_spec(
        schema=schema,
        source_url=source_url or (html_path or ""),
        page_title=page_title,
        fetch_mode=fetch_mode,
        text=text,
        form_controls=controls,
        spec=spec,
    )

    return RenderedPortalArtifacts(
        problem_text=render_problem_text(harvested, spec),
        refs_text=render_refs_text(harvested, spec),
        harvest=harvested,
    )


def write_harvest_artifacts(
    artifacts: RenderedPortalArtifacts,
    out_dir: str,
    stem: str,
) -> RenderedPortalArtifacts:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    json_path = p / f"{stem}_portal_harvest.json"
    problem_path = p / f"{stem}_portal_problem.txt"
    refs_path = p / f"{stem}_portal_refs.txt"

    json_path.write_text(
        json.dumps(artifacts.harvest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    problem_path.write_text(artifacts.problem_text, encoding="utf-8")
    refs_path.write_text(artifacts.refs_text, encoding="utf-8")

    artifacts.json_path = str(json_path)
    artifacts.problem_path = str(problem_path)
    artifacts.refs_path = str(refs_path)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest tax portal fields and render prompt-ready text.")
    parser.add_argument("--schema", required=True, help="Schema name, e.g. business_income")
    parser.add_argument("--url", default=None, help="Live portal URL")
    parser.add_argument("--html", default=None, help="Local HTML snapshot path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--stem", default=None, help="Output filename stem; default = schema")
    parser.add_argument("--timeout-ms", type=int, default=30000, help="Browser timeout in ms")
    args = parser.parse_args()

    artifacts = harvest_and_render_portal(
        schema=args.schema,
        url=args.url,
        html_path=args.html,
        timeout_ms=args.timeout_ms,
    )

    stem = args.stem or args.schema
    artifacts = write_harvest_artifacts(
        artifacts=artifacts,
        out_dir=args.out_dir,
        stem=stem,
    )

    print(f"[OK] schema={args.schema}")
    print(f"[OK] wrote: {artifacts.json_path}")
    print(f"[OK] wrote: {artifacts.problem_path}")
    print(f"[OK] wrote: {artifacts.refs_path}")


if __name__ == "__main__":
    main()