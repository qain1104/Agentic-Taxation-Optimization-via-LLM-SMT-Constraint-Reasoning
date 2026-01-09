# Agentic Taxation Optimization via LLM SMT-Constraint Reasoning — Artifact

This repository contains the artifact for the ICSE 2026 SEIP paper:

> **Agentic Taxation Optimization via LLM SMT-Constraint Reasoning**  
> _ICSE 2026 — Software Engineering in Practice (SEIP)_

The artifact includes the full implementation of the agentic tax optimization system, together with the data and scripts needed to reproduce the main experimental results (RQ1, RQ2, and RQ3).

> Language Note (Chinese-Only Artifact Interfaces)

Important: This artifact is Chinese-first.

The interactive system UI, example prompts, RQ1/RQ2/RQ3 task descriptions, and the LLM baseline prompts used in our experiments were written and executed in Traditional Chinese.

All runtime interaction with the agentic service (both Gradio UI and the FastAPI /run endpoint) expects Chinese natural-language inputs for user queries (unless you directly call a calculator tool with structured parameters).

What this means for artifact evaluators

If you are not fluent in Chinese, you can still reproduce our experiments by translating the provided Chinese prompts/tasks into English (or your preferred language).
However, translation is performed by the evaluator and is not included as part of this artifact, and translation choices may introduce minor semantic differences (e.g., tax terms, constraint wording, or intent strength such as “must” vs “should”).

Recommended translation practice

To reduce ambiguity and preserve reproducibility:

Translate conservatively (literal over creative). Keep numbers, entity names, and constraint bounds unchanged.

Do not paraphrase tax/legal terms unless necessary. If unsure, keep the Chinese term and add an English gloss, e.g.,
“標準扣除額 (standard deduction)”, “薪資所得 (salary income)”.

Keep an explicit record of translation:

Save the translated prompt alongside the original Chinese prompt (e.g., task_07.zh.txt + task_07.en.txt).

When reporting results, cite which version was used.

Optional helper (out of scope for evaluation)

Evaluators may use any translation tool (human or machine translation) to understand prompts and reports, but we do not require or provide a specific translation pipeline for badge evaluation. The artifact’s primary claim is that the system and baselines were evaluated under Chinese prompts, matching the original deployment context.
---

## Badges & Scope

We intend this artifact to qualify for the following ICSE / ACM badges:

- **Artifacts Available**  
  The full artifact (source code, data, and scripts) will be archived on a long-term repository (e.g., Zenodo) and linked from the camera-ready paper.

- **Artifacts Evaluated — Reusable**  
  The artifact is documented and structured to support reuse and extension beyond the paper’s experiments, e.g., adapting the pipeline to other tax regimes or constraint problems.

**Provenance**

- Paper: _Agentic Taxation Optimization via LLM SMT-Constraint Reasoning_ (ICSE 2026 SEIP).  
- Preprint: <!-- TODO: add arXiv / institutional link if available -->  
- Archived artifact DOI: https://doi.org/10.5281/zenodo.18182268

---

## Repository Layout

At the top level, the repository is organized as follows:

```text
.
├── Readme.md
└── source_code
    ├── Dockerfile
    ├── Readme.md
    ├── __pycache__
    ├── agents
    │   ├──  __init__.py
    │   ├── app_gradio.py
    │   ├── integrations
    │   │   ├── __pycache__
    │   │   │   └── fintax_api.cpython-311.pyc
    │   │   └── fintax_api.py
    │   ├── logs
    │   │   └── tax_app.log
    │   ├── lt_memory.json
    │   ├── multi_agent_tax_system.py
    │   ├── parsers
    │   │   ├── cargo_nl_parser.py
    │   │   ├── futures_nl_parser.py
    │   │   ├── nvat_nl_parser.py
    │   │   ├── securities_nl_parser.py
    │   │   ├── special_goods_nl_parser.py
    │   │   ├── ta_nl_parser.py
    │   │   └── vat_nl_parser.py
    │   ├── rag
    │   │   ├── README.md
    │   │   ├── chroma
    │   │   ├── ingest_pdfs.py
    │   │   ├── ingest_urls.py
    │   │   ├── pdfs
    │   │   │   ├── 114年貨物稅節稅手冊(PDF檔).pdf
    │   │   │   ├── 114年營利事業所得稅-節稅手冊.pdf
    │   │   │   ├── Readme.md
    │   │   │   ├── 國稅節稅手冊.pdf
    │   │   │   ├── 菸酒稅節稅手冊(PDF).pdf
    │   │   │   ├── 營業稅節稅手冊.pdf
    │   │   │   ├── 期貨交易稅節稅手冊_(1).pdf
    │   │   │   └── 證券交易稅節稅手冊_(1).pdf
    │   │   ├── test_search.py
    │   │   ├── tools
    │   │   │   └── clear_index.py
    │   │   └── urls.txt
    │   ├── report_renderer.py
    │   ├── reports
    │   │   └── last_run
    │   └── tools_registry.py
    ├── app.py
    ├── code_synthesis
    │   ├── Readme.md
    │   ├── chroma
    │   ├── generated_tax_solver.py
    │   ├── inputs
    │   │   ├── business_income.txt
    │   │   ├── business_tax.txt
    │   │   ├── cargo_tax.txt
    │   │   ├── estate_input.txt
    │   │   ├── foreign_income_tax.txt
    │   │   ├── gift_tax_input.txt
    │   │   ├── income_tax.txt
    │   │   ├── security_futures.txt
    │   │   ├── special_goods_services.txt
    │   │   └── ta_tax.txt
    │   ├── json_and_csv
    │   │   ├── all_laws.csv
    │   │   └── all_laws.json
    │   ├── protal_samples
    │   │   └── income_tax_samples.json ...
    │   ├── refs
    │   │   └── 114_numbers.txt
    │   ├── runs
    │   ├── tax_agent_pipeline.py
    │   └── taxrag
    │       ├── __init__.py
    │       ├── __main__.py
    │       ├── chroma_store.py
    │       ├── config.py
    │       ├── documents
    │       │   ├── 所得稅法.pdf
    │       │   ├── 菸酒稅法.pdf
    │       │   ├── 期貨交易法.pdf
    │       │   ├── 證券交易法.pdf
    │       │   ├── 貨物稅條例.pdf
    │       │   ├── 期貨交易稅條例.pdf
    │       │   ├── 證券交易稅條例.pdf
    │       │   ├── 遺產及贈與稅法.pdf
    │       │   ├── 所得基本稅額條例.pdf
    │       │   ├── 所得稅法施行細則.pdf
    │       │   ├── 期貨交易法施行細則.pdf
    │       │   ├── 證券交易法施行細則.pdf
    │       │   ├── 特種貨物及勞務稅條例.pdf
    │       │   ├── 營利事業所得稅查核準則.pdf
    │       │   ├── 遺產及贈與稅法施行細則.pdf
    │       │   ├── 加值型及非加值型營業稅法.pdf
    │       │   ├── 特種貨物及勞務稅條例施行細則.pdf
    │       │   ├── 加值型及非加值型營業稅法施行細則.pdf
    │       │   └── 特種貨物及勞務稅稅課收入分配及運用辦法.pdf
    │       ├── pdf_ingest.py
    │       ├── readme.md
    │       ├── requirements.txt
    │       ├── structure.py
    │       └── ui.py
    ├── docker-compose.yml
    ├── json_and_csv
    │   ├── all_laws.csv
    │   └── all_laws.json
    ├── requirements.txt
    ├── selenium_test_records
    │   ├── 01.綜合所得稅income_tax.ipynb
    │   ├── 02.外僑稅額試算_foregin_income_tax.ipynb
    │   ├── 03.營利事業所得稅_enterprise_income.ipynb
    │   ├── 04.營業稅_business_tax.ipynb
    │   ├── 05.貨物稅_cargo_tax.ipynb
    │   ├── 06.菸酒稅_ta_tax.ipynb
    │   ├── 07.遺產稅_estate_tax.ipynb
    │   ├── 08.贈與稅_gift_tax.ipynb
    │   ├── 09.證交期交稅_security_futures.ipynb
    │   ├── 10.特種貨物及勞務稅_special_services_goods.ipynb
    │   └── Readme.md
    └── tax_calculators
        ├── __init__.py
        ├── business_income_tax.py
        ├── cargo_tax.py
        ├── constraint_utils.py
        ├── estate_tax.py
        ├── foreigner_income_tax.py
        ├── gift_tax.py
        ├── high_consumption_goods_and_services_tax.py
        ├── income_tax.py
        ├── main.py
        ├── sale_tax.py
        ├── securities_and_futures_transaction_tax.py
        ├── special_tax.py
        ├── tax_calculator.py
        ├── tobacco_alcohol_tax.py
        └── util.py 
```

### `source_code/`

This directory contains the implementation of the agentic tax optimization service and all runtime configuration:

- `agents/`  
  - Multi-agent orchestration and tool registry (Caller / Constraint / Execute / Reasoning).  
  - Frontends such as:
    - `app_gradio.py` — Gradio UI (interactive demo).
  - Integrations in `agents/integrations/` (e.g., FIN backend webhook).

- `tax_calculators/`  
  - Per-tax-type calculators, each with its own generated Z3 SMT constraint modules, e.g.:
    - `income_tax.py`
    - `business_income_tax.py`
    - `gift_tax.py`
    - `tobacco_alcohol_tax.py`
    - …and others.  
  - Shared helpers such as `tax_calculator.py`, `constraint_utils.py`, `util.py`.

- `parsers/`  
  - RAG utilities and the report renderer used by the Reasoning agent.

- `reports/`  
  - Report rendering templates and, at runtime, the directory where the latest reports are written under `reports/last_run/` (see below).

- `logs/`  
  - Runtime log directory. The main rotating log file is `logs/tax_app.log` inside the container (or local directory when running locally).

- `Dockerfile`, `docker-compose.yml`  
  - Container and compose files **live inside `source_code/`**.  
  - Used to build and run the system as a self-contained service for this artifact.

- `requirements.txt`  
  - Python dependencies for running the backend without Docker.

- `Readme.md` (inside `source_code/`)  
  - Original system-level notes from the development repository (left largely unchanged).

> If you are only interested in using the system as a service, `source_code/` together with the Docker configuration is all you need.

# `source_code/code_synthesis`

Artifacts related to **methodology: code synthesis layer**:

- **Legal Clause Retrieval & Tax Code Generation**:  
  This directory contains the core functionality for retrieving tax law clauses from a Chroma vector store and synthesizing tax code based on the retrieved clauses using advanced LLMs.

- **PDF Ingestion Scripts**:  
  Scripts for ingesting legal documents (such as tax law PDFs) and converting them into a format suitable for indexing and retrieval in the Chroma vector store.

- **Chroma Integration**:  
  Files responsible for building the Chroma store with legal documents, as well as querying the stored clauses during the code generation process.

- **Code Synthesis Pipeline**:  
  A pipeline that generates and repairs tax computation code, using reasoning techniques to ensure accuracy and compliance with legal frameworks.

- **`README.md`**:  
  A comprehensive guide describing the steps and required dependencies to execute the pipeline, including how to ingest documents, build the Chroma store, and run the code generation pipeline.

- **Metrics Collection**:  
  Scripts and outputs used for evaluating the effectiveness of the generated tax computation code, including performance metrics and comparison results.

This directory is crucial for understanding the **code synthesis methodology** and reproducing the experiments.


### `source_code/selenium_test_records`

Artifacts related to **RQ2: Generalization to Unseen Inputs / Constraint Code Accuracy**:

- Random test-case generators that sample inputs within each portal field’s domain while respecting UI guards (e.g., spouse fields only when `married=true`, valid ranges/enums, precision constraints).
- Scripts to run Monte-Carlo validation by comparing our synthesized SMT solvers against the official MoF eTax calculators on large batches of unseen cases per regime.
- Collected mismatch logs (if any), per-regime accuracy summaries, and scripts to regenerate the accuracy numbers reported in the paper (solver = 100% vs GPT-5-level prompt-only baseline).
- A dedicated `source_code/selenium_test_records/Readme.md` describing the exact commands to reproduce the RQ2 experiments.

### `rq3/`

Artifacts related to **RQ3: Optimizing Real-World Tax Decisions**:

- The 20 natural-language tax-planning tasks across ten Taiwanese tax regimes used in the paper (e.g., minimizing consolidated income tax under constraints, maximizing purchasable quantity under a tax cap).
- Prompt templates and evaluation harnesses that translate NL tasks into our symbolic optimization calls (LLM + Z3 Optimize) and into the GPT-5-level baseline with Chain-of-Thought and browsing.
- Result files with per-task optimality and latency measurements, plus scripts to recompute the summary table comparing our system against the LLM baseline.
- A dedicated `rq3/README.md` describing how to run both systems and reproduce the RQ3 results end-to-end.

---

## Requirements

### Hardware

- CPU: any modern x86-64 CPU; 4+ cores recommended.
- RAM: **8 GB** minimum; **16 GB** recommended for larger batches.
- Disk: at least **5 GB** free (Docker image + logs + reports).
- GPU: **not required** for the SMT/agentic backend.  
  LLM calls are made via external APIs; no local GPU inference is performed.

### Software

You can run the artifact in two ways:

1. **Docker (recommended)**
   - Docker Engine ≥ 24.x
   - Docker Compose plugin

2. **Local Python environment**
   - Python ≥ 3.10
   - `pip` and ability to install standard scientific / web packages.

### External Services / Secrets

The system uses remote LLM and (optionally) integration endpoints:

- An LLM API key (e.g., `OPENAI_API_KEY`).  
- Optional: internal URLs for the tax portal and FIN backend (`FIN_BACKEND_BASEURL`, etc.).

For artifact evaluation, we provide configuration presets that:

- Disable calls to production government systems.  
- Either use the LLM API directly or rely on cached responses / stubs when no key is available (see the `rq*` READMEs).

The exact environment variables are documented in:

- `source_code/Readme.md`, and  
- `.env.example` next to `source_code/docker-compose.yml` (if provided).

---

## Setup

### Option A: Docker (recommended path)

1. **Clone the repository**

```bash
git clone https://github.com/<ORG>/<REPO>.git   # TODO: replace with real URL
cd <REPO>                                      # TODO: replace with real folder name
```

2. **Enter `source_code/`**

```bash
cd source_code
```

3. **Create a `.env` file**

If a `.env.example` is provided:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```text
OPENAI_API_KEY=your-api-key-here
# FIN_BACKEND_BASEURL=http://example-fin-backend:6677
# TAX_LOG_DIR=/app/logs
# (Add other variables as needed; see source_code/Readme.md)
```

4. **Build and start the backend**

```bash
docker compose build
docker compose up
```

This will:

- Build a container image for the agentic backend from `source_code/`.  
- Start the service exposing its UI / API on port **32770** (as configured in `docker-compose.yml`).

5. **Basic smoke test**

- If the Gradio UI is enabled as the container entrypoint, open a browser:  

  http://localhost:32770

  You should see the UI.

- If the FastAPI HTTP API is enabled, you can also issue a health check:

  ```bash
  curl http://localhost:32770/health
  ```

  Expected output:

  ```json
  { "status": "ok", "time": "..." }
  ```

This confirms that the containerized artifact is installed and running.

---

### Option B: Local Python install (alternative)

> This path is provided for users who prefer not to use Docker.  
> Docker remains the recommended option for artifact evaluation.

1. **Create and activate a virtual environment**

```bash
cd source_code
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set environment variables**

```bash
export OPENAI_API_KEY=your-api-key-here
# export TAX_LOG_DIR=./logs
# export FIN_BACKEND_BASEURL=http://example-fin-backend:6677
# (Others as required)
```

4. **Run the backend**

Choose one of the entrypoints:

- **Gradio UI**

  ```bash
  python agents/app_gradio.py
  ```

- **HTTP API (FastAPI)**

  ```bash
  python agents/app_fastapi.py
  ```

The service will listen on the port configured in the corresponding app (by default 32770).

---

## Basic API Usage Example

Once the HTTP API is running (via Docker or `app_fastapi.py`), you can interact with the REST endpoints.

### 1. Health check

```bash
curl http://localhost:32770/health
```

### 2. Single optimization run (`/run`)

```bash
curl -X POST http://localhost:32770/run \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "income_tax",
    "user_params": {
      "is_married": false,
      "salary_income": 1200000,
      "other_income": 0,
      "standard_deduction": true
    },
    "constraints": {},
    "free_vars": [],
    "op": "minimize",
    "budget_tax": null
  }'
```

Example response (simplified):

```json
{
  "status": "ok",
  "kpi": {
    "tax_before": 123456,
    "tax_after": 98765
  },
  "final_params": { "...": "..." },
  "diff": { "...": "..." },
  "report_md": "## Tax optimization report\n..."
}
```

### 3. Export latest report (`/export`)

```bash
curl -X POST http://localhost:32770/export \
  -H "Content-Type: application/json" \
  -d '{
    "format": "both",
    "push_to_fin": false
  }'
```

The response includes:

- `title` — report title  
- `md` — Markdown report content  
- `json` — JSON-structured report  
- `delivery` — delivery status if `push_to_fin=true`

At the file-system level, the latest report and delivery logs are also stored under:

```text
source_code/reports/last_run/
    ├── last.md
    ├── last.json
    ├── last.sent.log
    └── last.sent.response.json
```

---