from __future__ import annotations
import logging, re
from pathlib import Path
from typing import Dict, Any, List, Optional
import pdfplumber

logger = logging.getLogger(__name__)

_NOISE_PATTERNS = [
    r"^中華民國.*年.*月.*日.*$",
    r"^（?法規名稱：.*$",
    r"^法規名稱：.*$",
    r"^法規類別：.*$",
    r"^立法理由.*$",
    r"^修正.*條文.*$",
    r"^（?修正日期：.*$",
    r"^附件.*$",
    r"^第\s*\d+\s*頁\s*/\s*共\s*\d+\s*頁.*$",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS))

def extract_text_pdfplumber(pdf_file: str) -> str:
    parts: List[str] = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
            else:
                logger.debug("No text extracted on page %s of %s", i, pdf_file)
    return "\n".join(parts)

def clean_lines(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.splitlines():
        s = re.sub(r"\s+", " ", raw.strip())
        if not s:
            continue
        if _NOISE_RE.match(s):
            continue
        out.append(s)
    return out

_ARTICLE_RE = re.compile(r"^第\s*([0-9]+(?:\s*-\s*[0-9]+)?)\s*條\b")
_CLAUSE_RE  = re.compile(r"^(\d{1,2})\s*[\.、)]\s*(.*)$|^(\d{1,2})\s+(.*)$")
_SECTION_RE = re.compile(r"^第\s*[一二三四五六七八九十百千萬億]+\s*(章|節|款|目)\b")

def split_articles_and_clauses(lines: List[str]) -> Dict[str, Any]:
    articles: Dict[str, Any] = {}
    current_article: Optional[str] = None
    current_clause_idx: Optional[int] = None

    for line in lines:
        if _SECTION_RE.match(line):
            continue

        m = _ARTICLE_RE.match(line)
        if m:
            art_id = re.sub(r"\s+", "", m.group(1))
            current_article = art_id
            current_clause_idx = None
            articles[current_article] = {"title": line, "content": "", "clauses": []}
            continue

        if not current_article:
            continue

        cm = _CLAUSE_RE.match(line)
        if cm:
            clause_text = (cm.group(2) or cm.group(4) or "").strip()
            articles[current_article]["clauses"].append(clause_text)
            current_clause_idx = len(articles[current_article]["clauses"]) - 1
            continue

        # continuation
        if current_clause_idx is not None:
            prev = articles[current_article]["clauses"][current_clause_idx]
            articles[current_article]["clauses"][current_clause_idx] = re.sub(
                r"\s+", " ", (prev + " " + line).strip()
            )
        else:
            prev = articles[current_article]["content"]
            articles[current_article]["content"] = re.sub(
                r"\s+", " ", (prev + " " + line).strip()
            )

    return articles

def infer_law_name_from_filename(pdf_path: Path) -> str:
    return pdf_path.stem.replace("_", " ").strip()

def ingest_pdf_dir(pdf_dir: str) -> List[Dict[str, Any]]:
    p = Path(pdf_dir).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"PDF dir not found: {p}")
    pdfs = sorted([x for x in p.iterdir() if x.suffix.lower() == ".pdf"])
    if not pdfs:
        raise RuntimeError(f"No PDF files found under: {p}")

    laws: List[Dict[str, Any]] = []
    for pdf in pdfs:
        logger.info("Ingesting %s", pdf.name)
        text = extract_text_pdfplumber(str(pdf))
        lines = clean_lines(text)
        laws.append({
            "law_name": infer_law_name_from_filename(pdf),
            "pdf_file": str(pdf),
            "articles": split_articles_and_clauses(lines),
        })
    return laws
