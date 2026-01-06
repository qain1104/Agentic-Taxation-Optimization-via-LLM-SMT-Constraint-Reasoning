from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from .pdf_ingest import ingest_pdf_dir
from .structure import build_all_laws_payload
from .chroma_store import upsert_all, retrieve_clauses_chroma
from .ui import launch_ui

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def cmd_ingest(args):
    laws = ingest_pdf_dir(args.pdf_dir)
    payload = build_all_laws_payload(laws)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "all_laws.json"
    out_csv  = out_dir / "all_laws.csv"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    import pandas as pd
    rows = []
    for law in payload.get("laws", []):
        for article_id, ad in (law.get("articles", {}) or {}).items():
            rows.append({
                "law_name": law.get("law_name",""),
                "pdf_file": law.get("pdf_file",""),
                "article_id": article_id,
                "title": (ad or {}).get("title",""),
                "content": (ad or {}).get("content",""),
                "clauses_joined": " | ".join([(str(c)) for c in ((ad or {}).get("clauses", []) or [])]),
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("Wrote %s and %s", out_json, out_csv)

def cmd_build(args):
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    n = upsert_all(
        all_laws=data,
        persist_dir=args.chroma_dir,
        collection_name=args.collection,
        embed_model=args.embed_model,
        chunk_mode=args.chunk_mode,
        max_chars=args.max_chars,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    logger.info("Chroma upsert done: %s documents", n)

def cmd_query(args):
    print(retrieve_clauses_chroma(
        query=args.q,
        persist_dir=args.chroma_dir,
        collection_name=args.collection,
        k=args.k,
        embed_model=args.embed_model,
    ))

def cmd_ui(args):
    launch_ui(
        persist_dir=args.chroma_dir,
        collection_name=args.collection,
        embed_model=args.embed_model,
        k=args.k,
    )

def main():
    p = argparse.ArgumentParser("taxrag")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("ingest-pdfs")
    p1.add_argument("--pdf-dir", required=True)
    p1.add_argument("--out-dir", default="json_and_csv")
    p1.set_defaults(func=cmd_ingest)

    p2 = sub.add_parser("build-chroma")
    p2.add_argument("--json", required=True)
    p2.add_argument("--chroma-dir", default="chroma")
    p2.add_argument("--collection", default="laws_collection")
    p2.add_argument("--embed-model", default="text-embedding-3-large")
    p2.add_argument("--chunk-mode", choices=["clause","article"], default="clause")
    p2.add_argument("--max-chars", type=int, default=1800)
    p2.add_argument("--overlap", type=int, default=120)
    p2.add_argument("--batch-size", type=int, default=128)
    p2.set_defaults(func=cmd_build)

    p3 = sub.add_parser("query")
    p3.add_argument("--chroma-dir", default="chroma")
    p3.add_argument("--collection", default="laws_collection")
    p3.add_argument("--embed-model", default="text-embedding-3-large")
    p3.add_argument("--k", type=int, default=15)
    p3.add_argument("--q", required=True)
    p3.set_defaults(func=cmd_query)

    p4 = sub.add_parser("ui")
    p4.add_argument("--chroma-dir", default="chroma")
    p4.add_argument("--collection", default="laws_collection")
    p4.add_argument("--embed-model", default="text-embedding-3-large")
    p4.add_argument("--k", type=int, default=20)
    p4.set_defaults(func=cmd_ui)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
