from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Sequence

from .models import BranchSpec, FieldSpec, FormulaSpec, HarvestedPortal


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_visible_lines(text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def _find_matches(lines: Sequence[str], needle: str) -> List[str]:
    norm = re.sub(r"\s+", "", needle)
    hits = []
    for ln in lines:
        if norm and norm in re.sub(r"\s+", "", ln):
            hits.append(ln)
    return hits


def _field_found_in_controls(label: str, controls: Sequence[Any]) -> bool:
    label_norm = re.sub(r"\s+", "", label)
    for c in controls:
        hay = " ".join([
            getattr(c, "name", ""),
            getattr(c, "element_id", ""),
            getattr(c, "placeholder", ""),
            getattr(c, "aria_label", ""),
            getattr(c, "title", ""),
            getattr(c, "text_nearby", ""),
        ])
        if label_norm and label_norm in re.sub(r"\s+", "", hay):
            return True
    return False


def harvest_with_spec(*, schema: str, source_url: str, page_title: str, fetch_mode: str, text: str, form_controls: Sequence[Any], spec: Dict[str, Any]) -> HarvestedPortal:
    lines = normalize_visible_lines(text)
    fields: List[FieldSpec] = []
    for label, key in spec.get("field_aliases", {}).items():
        matches = _find_matches(lines, label)
        fields.append(
            FieldSpec(
                label=label,
                normalized_key=key,
                found_in_text=bool(matches),
                found_in_form_controls=_field_found_in_controls(label, form_controls),
                raw_matches=matches[:5],
            )
        )

    formulas: List[FormulaSpec] = []
    seen_formula = set()
    for hint in spec.get("formula_hints", []):
        matches = _find_matches(lines, hint) or [hint]
        for m in matches[:1]:
            if m not in seen_formula:
                formulas.append(FormulaSpec(text=m, source="hint" if m == hint else "text"))
                seen_formula.add(m)

    notes: List[str] = []
    for line in lines:
        for hint in spec.get("note_hints", []):
            if re.sub(r"\s+", "", hint) in re.sub(r"\s+", "", line):
                if line not in notes:
                    notes.append(line)
                break

    branches: List[BranchSpec] = []
    for hint in spec.get("branch_hints", []):
        branch_lines = [ln for ln in lines if re.sub(r"\s+", "", hint) in re.sub(r"\s+", "", ln)]
        if branch_lines:
            branches.append(BranchSpec(heading=hint, lines=branch_lines[:10]))

    extra = []
    for ln in lines:
        if any(x in ln for x in ["=", "稅率", "免稅", "應納", "課稅"]):
            if ln not in notes and ln not in [f.text for f in formulas]:
                extra.append(ln)

    excerpt = "\n".join(lines[:80])
    return HarvestedPortal(
        schema=schema,
        source_url=source_url,
        title=spec.get("title", schema),
        fetched_at=_now_iso(),
        fetch_mode=fetch_mode,
        page_title=page_title,
        visible_lines=lines,
        form_controls=list(form_controls),
        fields=fields,
        formulas=formulas,
        branches=branches,
        notes=notes[:50],
        extra_lines=extra[:80],
        raw_text_excerpt=excerpt,
    )
