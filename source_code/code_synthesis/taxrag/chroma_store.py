from __future__ import annotations
import hashlib, logging
from typing import Any, Dict, List, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from .config import get_openai_key

logger = logging.getLogger(__name__)

def _stable_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()

def _chunk_text(text: str, max_chars: int = 1800, overlap: int = 120) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    out = []
    i = 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        out.append(text[i:j].strip())
        if j == len(text):
            break
        i = max(0, j - overlap)
    return [x for x in out if x]

def open_collection(persist_dir: str, collection_name: str, embed_model: str):
    key = get_openai_key()
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    ef = OpenAIEmbeddingFunction(api_key=key, model_name=embed_model)
    return client.get_or_create_collection(name=collection_name, embedding_function=ef)

def build_docs(all_laws: Dict[str, Any], chunk_mode: str, max_chars: int, overlap: int):
    docs, metas, ids = [], [], []
    for law in all_laws.get("laws", []):
        law_name = str(law.get("law_name","")).strip()
        pdf_file = str(law.get("pdf_file","")).strip()
        articles = law.get("articles", {}) or {}
        if not law_name or not isinstance(articles, dict):
            continue

        for article_id, ad in articles.items():
            if not isinstance(ad, dict):
                continue
            title = str(ad.get("title","")).strip()
            content = str(ad.get("content","")).strip()
            clauses = ad.get("clauses", []) or []

            if chunk_mode == "article":
                full = "\n".join([x for x in [title, content] if x])
                if clauses:
                    full += "\n" + "\n".join([str(c).strip() for c in clauses if str(c).strip()])
                for ci, chunk in enumerate(_chunk_text(full, max_chars, overlap)):
                    docs.append(chunk)
                    metas.append({"law_name": law_name, "article_id": str(article_id), "title": title, "pdf_file": pdf_file,
                                  "chunk_mode": "article", "chunk_index": ci})
                    ids.append(_stable_id(law_name, str(article_id), f"article:{ci}", chunk[:64]))

            else:  # clause
                if clauses:
                    for clause_idx, clause in enumerate(clauses):
                        clause_text = str(clause).strip()
                        if not clause_text:
                            continue
                        clause_full = "\n".join([x for x in [title, clause_text] if x])
                        for ci, chunk in enumerate(_chunk_text(clause_full, max_chars, overlap)):
                            docs.append(chunk)
                            metas.append({"law_name": law_name, "article_id": str(article_id), "title": title, "pdf_file": pdf_file,
                                          "chunk_mode": "clause", "clause_index": clause_idx, "chunk_index": ci})
                            ids.append(_stable_id(law_name, str(article_id), f"clause:{clause_idx}:{ci}", chunk[:64]))
                else:
                    full = "\n".join([x for x in [title, content] if x])
                    for ci, chunk in enumerate(_chunk_text(full, max_chars, overlap)):
                        docs.append(chunk)
                        metas.append({"law_name": law_name, "article_id": str(article_id), "title": title, "pdf_file": pdf_file,
                                      "chunk_mode": "content", "chunk_index": ci})
                        ids.append(_stable_id(law_name, str(article_id), f"content:{ci}", chunk[:64]))
    return docs, metas, ids

def upsert_all(all_laws: Dict[str, Any], persist_dir: str, collection_name: str,
              embed_model: str="text-embedding-3-large", chunk_mode: str="clause",
              max_chars: int=1800, overlap: int=120, batch_size: int=128) -> int:
    col = open_collection(persist_dir, collection_name, embed_model)
    docs, metas, ids = build_docs(all_laws, chunk_mode, max_chars, overlap)
    if not docs:
        logger.warning("No documents built from all_laws.json")
        return 0

    total = 0
    for i in range(0, len(docs), batch_size):
        bd, bm, bi = docs[i:i+batch_size], metas[i:i+batch_size], ids[i:i+batch_size]
        try:
            col.upsert(ids=bi, documents=bd, metadatas=bm)
        except Exception:
            col.add(ids=bi, documents=bd, metadatas=bm)
        total += len(bd)
        logger.info("Upserted %s/%s", total, len(docs))
    return total

def retrieve_clauses_chroma(query: str, persist_dir: str, collection_name: str, k: int=15,
                           embed_model: str="text-embedding-3-large") -> str:
    col = open_collection(persist_dir, collection_name, embed_model)
    res = col.query(query_texts=[query], n_results=k, include=["documents","metadatas"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    out = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        head = []
        if meta.get("law_name"): head.append(str(meta["law_name"]))
        if meta.get("article_id"): head.append(f"第{meta['article_id']}條")
        if meta.get("title"): head.append(str(meta["title"]))
        header = " / ".join(head).strip()
        out.append(f"[{i+1}] {header}\n{doc}" if header else f"[{i+1}]\n{doc}")
    return "\n\n".join(out).strip()
