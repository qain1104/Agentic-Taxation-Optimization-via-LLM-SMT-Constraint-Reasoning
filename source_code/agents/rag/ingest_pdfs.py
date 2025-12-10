#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將資料夾中的 PDF 手冊切片後寫入 Chroma。
用法：
    python rag/ingest_pdfs.py --pdf_dir rag/pdfs --chunk_size 1200 --chunk_overlap 150
"""
import os, re, argparse, json
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

EMBED_MODEL = "text-embedding-3-small"
COLLECTION  = "tax_handbook"
CHROMA_DIR  = "chroma"

def infer_meta_from_name(name: str):
    base = os.path.basename(name)
    title = re.sub(r"\.pdf$", "", base, flags=re.I)
    # 嘗試從檔名抓章節/主題/年分
    m_year = re.search(r"(20\d{2}|19\d{2})", title)
    year = int(m_year.group(1)) if m_year else None
    return {"title": title, "year": year}

def main(pdf_dir: str, chunk_size: int, chunk_overlap: int):
    if not os.path.isdir(pdf_dir):
        raise SystemExit(f"找不到資料夾：{pdf_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = Chroma(collection_name=COLLECTION, persist_directory=CHROMA_DIR, embedding_function=embeddings)

    files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    files.sort()
    print(f"[INFO] 發現 PDF {len(files)} 筆")

    total_chunks = 0
    for path in files:
        meta = infer_meta_from_name(path)
        print(f"[PDF] {meta['title']}")

        loader = PyPDFLoader(path)
        docs = loader.load()
        merged = "".join([d.page_content for d in docs])
        if not merged.strip():
            print(f"[WARN] 空內容：{path}")
            continue

        splits = splitter.split_text(merged)
        # 包成 LangChain Documents（帶 metadata）
        from langchain.schema import Document
        documents = [Document(page_content=s, metadata=meta) for s in splits]

        db.add_documents(documents)
        total_chunks += len(documents)

    db.persist()
    print(f"[DONE] 已寫入 {total_chunks} 個分段到 Chroma『{COLLECTION}』於 {CHROMA_DIR}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", default="pdfs")
    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    args = ap.parse_args()
    main(args.pdf_dir, args.chunk_size, args.chunk_overlap)
