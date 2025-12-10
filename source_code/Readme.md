# 稅法 AI Agent 系統 README

本文件為本專案的「**使用與開發說明文件**」，提供開發者與使用者快速了解系統目的、啟動方式與主要程式結構。

---

## 1. 專案簡介

「稅法 AI Agent」是一套結合 **大型語言模型（LLM）** 與 **多代理（Multi-Agent）** 架構的稅務試算與分析系統。

透過自然語言與表單輸入條件，系統可協助使用者完成以下工作：

* 各類稅別的 **稅額試算**
* 在限制條件下進行 **最佳化**（例如在稅額預算內最大化產量／銷售量）
* 設定並管理 **條件限制（constraints）與自由變數（free vars）**
* 產出具可讀性與可稽核性的 **報告（報告輸出）**

### 支援稅別 (以 `tools_registry.py` 為準)

* 綜合所得稅（Income Tax）
* 營利事業所得稅（Business Income Tax）
* 遺產稅（Estate Tax）
* 贈與稅（Gift Tax）
* 貨物稅（Cargo Tax）
* 加值型與非加值型營業稅（VAT / NVAT）
* 特種貨物稅、菸酒稅等其他國稅相關稅目

> 📝 **RAG 支援：** 系統可搭配 **RAG（Retrieval-Augmented Generation）**，參考財政部「國稅節稅手冊」等官方文件內容，提供條文說明與節稅建議。

---

## 2. 🚀 快速開始（Quick Start）

### 2.1 本機執行流程（不使用 Docker）

1.  **下載專案程式碼**
    ```bash
    git clone <YOUR_REPO_URL> tax-ai-agent
    cd tax-ai-agent
    ```

2.  **建立虛擬環境**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate
    ```

3.  **安裝套件**
    ```bash
    pip install -r requirements.txt
    ```

4.  **設定環境變數**
    * 建立 `.env` 檔，**至少需設定 `OPENAI_API_KEY`**。

5.  **啟動 Gradio 介面**
    ```bash
    python app_gradio.py
    ```

---

## 3. ⚙️ 環境變數與設定說明

專案建議使用 `.env` 檔集中管理環境變數，並由程式在啟動時讀取。

### 3.1 必填參數

| 參數名稱 | 說明 | 用途 |
| :--- | :--- | :--- |
| **`OPENAI_API_KEY`** | OpenAI API 使用所需之金鑰。 | 供各 AI Agent 呼叫 LLM 模型與 Embedding 模型。 |

### 3.2 RAG 相關目錄設定（如有啟用）

| 參數名稱 | 說明 | 用途 |
| :--- | :--- | :--- |
| `RAG_CHROMA_DIR` | 向量資料庫（例如 ChromaDB）的持久化存放目錄。 | 儲存節稅手冊等外部文件的向量索引。 |
| `RAG_COLLECTION` | RAG 使用的 collection 名稱。 | 用於區分不同文件來源或索引。 |
| `RAG_K` （選用）| 每次查詢回傳的文件片段數量（Top-K）。 | 影響 RAG 檢索的準確性。 |
| `RAG_MIN_SCORE` （選用）| 相似度得分門檻。 | 低於門檻的結果可視為無關，不予引用。 |

### 3.3 Log 與報告輸出設定

| 參數名稱 | 說明 | 用途 |
| :--- | :--- | :--- |
| `TAX_LOG_DIR` （選用）| Log 檔案輸出目錄。 | 紀錄系統執行流程與錯誤，便於除錯與稽核。 |
| 報告輸出路徑 | 用於存放最後一次運算結果報告（Markdown / JSON）。 | 常見為 `./reports/last_run` 等路徑，可由程式設定或環境變數控制。 |

### 3.4 其他常見參數（視實作而定）

| 參數名稱 | 說明 |
| :--- | :--- |
| `GRADIO_SERVER_PORT` | Gradio Web 服務的對外埠號（預設可為 **32770**）。 |
| 其他參數 | 與 SMT、最佳化或對外 API 整合相關的參數，可於程式註解或內部文件中補充。 |

---

## 4. 🧠 系統架構與檔案結構簡介

本專案採用前後端整合的 Python 架構，以 **Gradio** 提供 Web 介面，**多 Agent** 作為後端核心邏輯，並透過工具註冊表與稅務計算模組整合各稅別邏輯。

### 4.1 系統架構概念

| 模組 | 角色與核心功能 | 關鍵組件 |
| :--- | :--- | :--- |
| **前端與入口** (Frontend & Entry Point) | 接收使用者輸入，以圖文方式呈現結果。 | Gradio Web UI：文字輸入框、條件表單、結果顯示區。 |
| **多代理核心** (Multi-Agent Core) | 控制 Agent 之間資料流與狀態，將輸入轉為計算結果與報告。 | **CallerAgent**、**ConstraintAgent**、**ExecuteAgent**、**ReasoningAgent**。 |
| **工具註冊與稅務計算模組** (Tools & Tax Calculators) | 以統一介面管理不同稅別的計算函式、欄位規格與可用限制。 | 各稅別工具函式、`tools_registry.py`。 |
| **RAG 與外部文件支援** (RAG Support) | 將外部 PDF 轉為向量索引，供 Agent 查詢與引用。 | `ingest_pdfs.py`、向量資料庫（ChromaDB）。 |

### 4.2 關鍵檔案說明

| 檔案名稱 | 角色 | 功能說明 |
| :--- | :--- | :--- |
| **`app_gradio.py`** | 前端與入口 | 建立 Gradio 介面，負責接收使用者輸入並呼叫多代理核心。 |
| **`multi_agent_tax_system.py`** | 多代理核心 | 定義並協調各 Agent 運作，控制資料流，產出最終結果與報告。 |
| **`tools_registry.py`** | 工具註冊表 | 登記各稅別工具的名稱、模組、入口函式、**必要欄位** (`required_fields`) 與**條件欄位** (`constraint_fields`) 等資訊。 |
| **`ingest_pdfs.py`** | RAG 索引建立程式 | 讀取節稅手冊等 PDF 文件，解析、切塊，並將文本轉為向量寫入向量資料庫。 |
| **`docker-compose.yml`** | 部署相關設定 | 定義服務的建置方式、埠號映射（例如 `32770:32770`）、`.env` 檔與 Docker network 設定。 |
| **`Dockerfile`** | Docker 映像建置描述檔 | 指定基底映像、安裝相依套件、複製專案程式碼並指定啟動指令。 |


### 5. RAG說明：如需匯入新的PDF稅法 可以參考 agents/rag/README.md 