from __future__ import annotations

from typing import Dict, List

from .models import HarvestedPortal


def _field_line(label: str, final: bool = False) -> str:
    return f"{label} = {'??' if final else '_'}"


def render_linear_template(h: HarvestedPortal, spec: Dict) -> str:
    lines: List[str] = [f"{h.title}："]
    for f in h.formulas:
        lines.append(f.text)
        used = []
        for field in h.fields:
            if field.label in f.text:
                used.append(field.label)
        for label in used:
            lines.append(_field_line(label))
    if h.notes:
        lines.append("\n說明：")
        lines.extend(h.notes[:12])
    lines.append(_field_line("應納稅額(T)", final=True))
    return "\n".join(lines).strip()


def render_branch_template(h: HarvestedPortal, spec: Dict) -> str:
    g = spec.get("format_guidance", {})
    order = g.get("branch_order", [])
    lines: List[str] = [f"{h.title}：", "類別 = _", ""]
    for branch in order:
        lines.append(f"[若為{branch}，規則如下]" if not branch.startswith("若為") else branch)
        rel_formulas = [f.text for f in h.formulas if branch[:3] in f.text or not h.formulas]
        if rel_formulas:
            for ftxt in rel_formulas[:2]:
                lines.append(ftxt)
        else:
            lines.append("公式：? = ?")
        for field in h.fields:
            lines.append(_field_line(field.label))
        notes = [n for n in h.notes if branch[:2] in n or "稅率" in n]
        if notes:
            lines.extend(notes[:5])
        else:
            lines.append("稅率 / 例外說明：?")
        lines.append("")
    lines.append(_field_line("應納稅額(T)", final=True))
    return "\n".join(lines).strip()


def render_dispatch_template(h: HarvestedPortal, spec: Dict) -> str:
    g = spec.get("format_guidance", {})
    order = g.get("branch_order", [])
    lines: List[str] = [f"{h.title}：", "tax_item = _", ""]
    for branch in order:
        lines.append(f"[若 tax_item 為 {branch}]")
        if "證券" in branch:
            for label in ["證券成交價格", "每股發行價格", "股數"]:
                lines.append(_field_line(label))
            lines.append("稅率 = ?")
        elif "期貨" in branch:
            for label in ["契約金額", "口數"]:
                lines.append(_field_line(label))
            lines.append("稅率 = ?")
        branch_notes = [n for n in h.notes if ("證券" in n and "證券" in branch) or ("期貨" in n and "期貨" in branch)]
        lines.extend(branch_notes[:4])
        lines.append("")
    lines.append(_field_line("應納稅額(T)", final=True))
    return "\n".join(lines).strip()


def render_problem_text(h: HarvestedPortal, spec: Dict) -> str:
    style = spec.get("render_style")
    if style == "linear_equation":
        return render_linear_template(h, spec)
    if style == "branch_template":
        return render_branch_template(h, spec)
    if style == "dispatch_template":
        return render_dispatch_template(h, spec)
    raise KeyError(f"Unknown render style: {style}")


def render_refs_text(h: HarvestedPortal, spec: Dict) -> str:
    lines: List[str] = []
    lines.append(f"[Portal Harvest] schema={h.schema}")
    lines.append(f"title: {h.title}")
    lines.append(f"source_url: {h.source_url}")
    lines.append(f"fetch_mode: {h.fetch_mode}")
    lines.append(f"fetched_at: {h.fetched_at}")
    if h.page_title:
        lines.append(f"page_title: {h.page_title}")
    lines.append("")
    lines.append("[Format Guidance]")
    lines.append(spec.get("format_guidance", {}).get("summary", ""))
    lines.append("")
    lines.append("[Harvested Fields]")
    for f in h.fields:
        lines.append(
            f"- {f.label} -> {f.normalized_key} | text={f.found_in_text} | control={f.found_in_form_controls}"
        )
        for rm in f.raw_matches[:2]:
            lines.append(f"  match: {rm}")
    lines.append("")
    lines.append("[Harvested Formulas]")
    for f in h.formulas:
        lines.append(f"- {f.text} ({f.source})")
    lines.append("")
    lines.append("[Harvested Notes]")
    for n in h.notes[:20]:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("[Visible Text Excerpt]")
    lines.append(h.raw_text_excerpt or "(empty)")
    return "\n".join(lines).strip()
