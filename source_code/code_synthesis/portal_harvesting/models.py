from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FieldSpec:
    label: str
    normalized_key: str
    placeholder: str = "_"
    found_in_text: bool = False
    found_in_form_controls: bool = False
    raw_matches: List[str] = field(default_factory=list)


@dataclass
class FormulaSpec:
    text: str
    source: str = "text"


@dataclass
class BranchSpec:
    heading: str
    lines: List[str] = field(default_factory=list)


@dataclass
class FormControl:
    tag: str
    control_type: str
    name: str
    element_id: str
    placeholder: str
    aria_label: str
    title: str
    value: str
    text_nearby: str = ""


@dataclass
class HarvestedPortal:
    schema: str
    source_url: str
    title: str
    fetched_at: str
    fetch_mode: str
    page_title: str = ""
    visible_lines: List[str] = field(default_factory=list)
    form_controls: List[FormControl] = field(default_factory=list)
    fields: List[FieldSpec] = field(default_factory=list)
    formulas: List[FormulaSpec] = field(default_factory=list)
    branches: List[BranchSpec] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    extra_lines: List[str] = field(default_factory=list)
    raw_text_excerpt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RenderedPortalArtifacts:
    problem_text: str
    refs_text: str
    harvest: HarvestedPortal
    json_path: Optional[str] = None
    problem_path: Optional[str] = None
    refs_path: Optional[str] = None
