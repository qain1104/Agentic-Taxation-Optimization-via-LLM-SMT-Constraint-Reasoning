# RAG for 稅務節稅手冊 → Chroma
這是一個極簡流程，將 **PDF 手冊** 與（可選）**網頁文字** 轉成 Chroma 向量索引，供 `_early_condition_tips()` 檢索。

## 1) 安裝套件
```bash
pip install -U "langchain-community>=0.2" "langchain-openai>=0.1" chromadb pypdf tiktoken trafilatura python-dotenv
```

> 你已經在專案使用 `langchain_openai`、`Chroma`，版本建議如上。

## 2) 準備資料
把你那些 PDF 檔全部放到：`rag/pdfs/` 資料夾。  
（或修改 `ingest_pdfs.py` 的路徑參數，指向你的檔案夾）

若有網頁（純文字/條文），把網址逐行寫在：`rag/urls.txt`（可選）。

## 3) 設定 API Key
在專案根目錄放一個 `.env`：
```
OPENAI_API_KEY=sk-...
```

## 4) 產生索引
```bash
# 4.1 先清空舊索引（可選）
python rag/tools/clear_index.py

# 4.2 匯入 PDF
python rag/ingest_pdfs.py --pdf_dir rag/pdfs

# 4.3 匯入網址（可選）
python rag/ingest_urls.py --urls_file rag/urls.txt

# 4.4 用小工具測試查詢
python rag/test_search.py "教育支出 租金 扣除 上限 20 萬"
```

完成後，Chroma 會保存在 `rag/chroma/`，你的系統就能在 `_early_condition_tips()` 用：
```python
db = Chroma(collection_name="tax_handbook", persist_directory="rag/chroma", embedding_function=OpenAIEmbeddings(...))
docs = db.similarity_search("你的查詢", k=3)
```

---

## 檔案一覽
- `rag/ingest_pdfs.py`：掃描資料夾 PDF → 切片 → 寫入 Chroma
- `rag/ingest_urls.py`：抓網頁 → 文字清洗 → 切片 → 寫入 Chroma（可選）
- `rag/test_search.py`：快速查詢檢索效果
- `rag/tools/clear_index.py`：清空索引資料夾（避開版本衝突）

> **注意**：PDF 解析使用 `pypdf`，已足夠你目前的手冊；若遇到掃描影像 PDF，需要 OCR（另行加入 `pytesseract` + `pdfplumber`/`pymupdf`）。
