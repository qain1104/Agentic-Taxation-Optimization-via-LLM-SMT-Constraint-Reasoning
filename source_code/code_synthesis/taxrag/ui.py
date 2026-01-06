from __future__ import annotations
from typing import Tuple
import gradio as gr
from .chroma_store import open_collection

def query_collection(query: str, persist_dir: str, collection_name: str, k: int, embed_model: str) -> Tuple[str,str]:
    col = open_collection(persist_dir, collection_name, embed_model)
    res = col.query(query_texts=[query], n_results=k, include=["documents","metadatas"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    md, combined = [], []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        law = meta.get("law_name","未知法規")
        art = meta.get("article_id","")
        title = meta.get("title","")
        head = f"{law} 第{art}條 {title}".strip()
        md.append(f"### {i}. {head}\n\n{doc}\n\n---\n")
        combined.append(f"\n{doc}\n")
    return "\n".join(md), "\n".join(combined)

def launch_ui(persist_dir="chroma", collection_name="laws_collection", embed_model="text-embedding-3-large", k=20):
    with gr.Blocks() as demo:
        gr.Markdown("## 稅法 RAG Demo（Chroma）")
        q = gr.Textbox(label="輸入你的問題 / 情境")
        btn = gr.Button("搜尋")
        out = gr.Markdown()
        copy_box = gr.Textbox(label="可複製的法條內容", lines=16, interactive=False)

        btn.click(lambda x: query_collection(x, persist_dir, collection_name, k, embed_model),
                  inputs=q, outputs=[out, copy_box])
    demo.launch()
