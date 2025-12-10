#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抓網址內容 → 清洗 → 切片 → 寫入 Chroma。
用法：
    python rag/ingest_urls.py --urls_file rag/urls.txt --chunk_size 1200 --chunk_overlap 150
"""
import os, argparse, time, re
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()

import trafilatura
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

EMBED_MODEL = "text-embedding-3-small"
COLLECTION  = "tax_handbook"
CHROMA_DIR  = "rag/chroma"

def fetch(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
    return text

def normalize(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def main(urls_file: str, chunk_size: int, chunk_overlap: int):
    if not os.path.exists(urls_file):
        print(f"[WARN] 找不到 {urls_file}，略過。")
        return

    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = Chroma(collection_name=COLLECTION, persist_directory=CHROMA_DIR, embedding_function=embeddings)

    total_chunks = 0
    for u in urls:
        print(f"[URL] {u}")
        try:
            raw = fetch(u)
        except Exception as e:
            print(f"[ERR] {e}")
            continue
        if not raw.strip():
            print("[WARN] 空內容")
            continue

        txt = normalize(raw)
        splits = splitter.split_text(txt)

        md = {"title": urlparse(u).netloc, "source": u}
        docs = [Document(page_content=s, metadata=md) for s in splits]

        db.add_documents(docs)
        total_chunks += len(docs)
        time.sleep(0.5)  # 禮貌性間隔

    db.persist()
    print(f"[DONE] 網頁共寫入 {total_chunks} 個分段到 Chroma『{COLLECTION}』於 {CHROMA_DIR}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls_file", default="rag/urls.txt")
    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    args = ap.parse_args()
    main(args.urls_file, args.chunk_size, args.chunk_overlap)
