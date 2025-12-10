#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, json
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

EMBED_MODEL = "text-embedding-3-small"
COLLECTION  = "tax_handbook"
CHROMA_DIR  = "rag/chroma"

def main(q: str):
    db = Chroma(collection_name=COLLECTION, persist_directory=CHROMA_DIR,
                embedding_function=OpenAIEmbeddings(model=EMBED_MODEL))
    docs = db.similarity_search(q, k=3)
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        print(f"\n== Hit {i} ==")
        print(f"[Title] {md.get('title')}", f"[Year] {md.get('year')}", sep="  ")
        print(d.page_content[:800])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python rag/test_search.py \"查詢字串\"")
        sys.exit(0)
    main(sys.argv[1])
