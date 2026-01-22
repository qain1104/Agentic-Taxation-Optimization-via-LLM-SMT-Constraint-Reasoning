# Agentic Taxation Optimization via LLM SMT-Constraint Reasoning — ICSE 2026 Artifact (Paper #127)

**Zenodo DOI:** https://doi.org/10.5281/zenodo.18182268  
**Requested badge:** **Artifacts Evaluated – Functional** *(and Artifacts Available)*  
**Code license:** **MIT License** (Zenodo metadata set to MIT; `LICENSE` included in the archive)

---

## Release Notes: ICSE 2026 Artifact Evaluation Update

This release addresses the documentation and packaging requirements requested by the AE reviewers for Paper #127.
The primary focus is to ensure functional completeness and ease of execution for the **Artifacts Evaluated – Functional** badge.

### Major Changes & Fixes

- **[Documentation]** Added a comprehensive, step-by-step `README.md` covering system requirements, installation, and end-to-end execution.
- **[Testing]** Introduced a **Minimal Functional Test** workflow intended to complete within **≤10 minutes** *after the Docker image is built*.
- **[Environment]** Documented dependency configuration (Docker-first), Python base image version, and SMT solver usage.
- **[Transparency]** Documented expected outputs and added a troubleshooting section for common LLM API and runtime issues.
- **[Legal]** Included clear license information (MIT) in the root directory and Zenodo metadata.

**How to verify:** Follow the **Linear walkthrough** section below to run the end-to-end workflow and confirm outputs.

---

## What this artifact contains

This artifact contains a complete implementation of an **agentic taxation optimization** system that combines:

- **LLM orchestration** (for parsing, reasoning, and report drafting), and  
- **SMT constraint reasoning (Z3)** (for numeric constraint satisfaction / optimization)

to generate optimization results and reproduce the workflows described in the paper.

**Primary entry point:** a **Gradio UI** served by `agents/app_gradio.py` (inside the Docker image).  
**Included data:** the archive includes **author-generated logs** and **author-generated Selenium validation records** (RQ2). These are not third‑party datasets.

**Language note (Chinese-first):** the UI and prompts are Traditional Chinese-first. Evaluators can translate if needed.

---

## Quick Start (Minimal Functional Test)

This is the minimal end-to-end functional test intended for ICSE AE.  
It is UI-driven and should complete within **≤10 minutes** once the service is running.

### 0) Download & unzip (Zenodo)

1. Go to the Zenodo record (DOI above) and download the archive.
2. Unzip it and enter the extracted folder.

```bash
unzip *.zip
ls -1d */ | head -n 1
cd <EXTRACTED_ARTIFACT_DIR>
```

### 1) Configure environment variables

```bash
cd source_code
cp .env.example .env
```

Edit `.env` and set at least:

```text
OPENAI_API_KEY=your-api-key-here
```

**Internet access is required** (remote LLM API). Offline / no-API-key mode is not supported in this release.

### 2) Build and run

```bash
# First build may take ~10 minutes (downloads + dependency install)
docker compose build

# Run the service (foreground; recommended for first run)
docker compose up
```

The UI is exposed on:

- http://localhost:32770

### 3) Verify the service is alive (liveness check)

In a new terminal:

```bash
curl -sSf http://localhost:32770/openapi.json >/dev/null && echo "OK"
```

**Success criteria:** prints `OK`.  
*(Note: `/docs` may be disabled; `/openapi.json` is the canonical liveness check.)*

### 4) Run the UI “Income tax” test case (「下一步」流程)

Open the UI:

- http://localhost:32770

#### Step 0 — Select which tax to compute (required)

Before entering the 3-phase workflow, **tell the system which tax you want to compute** (e.g., income tax),
**or click the corresponding button on the right panel**.

Type and send:

> 我要算所得稅

#### Phase 1 — Tax scenario input (copy/paste)

Paste and send:

> 我是一名公司高管，年收400萬、太太在家當貴婦，我們有兩個小孩。

Then type and send:

> 下一步

#### Phase 2 — Customized condition input (copy/paste)

Paste and send:

> 我想嘗試調整一些配置，所得總和400萬。

Then type and send:

> 下一步

#### Phase 3 — Calculation & report generation

Wait for the system to generate an optimization result and a report.

**UI success criteria:**  
- a generated result/report is displayed in the UI (Markdown-like text is acceptable), and  
- logs and/or report artifacts are written under `source_code/logs/` and/or `source_code/reports/` (see below).

---

## Expected outputs (what to look for)

Because this artifact is UI-driven, filenames may vary by configuration.
We define an **authoritative verification procedure**:

After completing the UI test case once, run:

```bash
# show recent files (modified within the last ~10 minutes)
find source_code -maxdepth 5 -type f -mmin -10 | sort

# common output folders
find source_code -maxdepth 5 -type f | grep -E "reports|output|outputs|logs|artifacts" | sort | head -n 200
```

### Typical output locations

- **Logs:** `source_code/logs/` (e.g., `tax_app.log`)
- **Reports / artifacts:** `source_code/reports/` (report files and/or JSON records depending on configuration)

**Functional success criteria (files):**
- at least one file under `source_code/logs/` is updated during the run, and  
- at least one new/updated artifact appears under `source_code/reports/` (if report export is enabled in your configuration).

---

## System requirements

### Supported OS

- **Linux / macOS / Windows** (via Docker)

### Hardware (minimum)

- CPU: x86-64, 4+ cores recommended  
- RAM: 8 GB minimum (16 GB recommended for larger batches)  
- Disk: ≥ 5 GB free  
- GPU: not required

### Software

- **Docker (recommended for ICSE AE):** Docker Engine + Docker Compose plugin
- The Docker image is based on **`python:3.11-slim`** and includes Python dependencies.

### External tools / solvers

- **Z3:** the Docker image includes the Python package **`z3-solver`**. No separate system-wide Z3 install is required for the Docker path.

### External services / secrets

- A valid LLM API key (e.g., `OPENAI_API_KEY`) is required.
- Internet access is required to call the remote LLM API.

---

## Linear walkthrough (clean machine)

1. Install Docker + Docker Compose  
2. Download the Zenodo archive and unzip  
3. `cd <EXTRACTED_ARTIFACT_DIR>/source_code`  
4. `cp .env.example .env` and set `OPENAI_API_KEY`  
5. `docker compose build`  
6. `docker compose up`  
7. `curl -sSf http://localhost:32770/openapi.json >/dev/null && echo "OK"`  
8. Open `http://localhost:32770`  
9. Run the “Income tax” UI test case (Step 0 → Phase 1 → 下一步 → Phase 2 → 下一步 → Phase 3)  
10. Inspect outputs under `source_code/logs/` and `source_code/reports/` using the commands in **Expected outputs**

---

## Minimal Functional Test suite (scripted helper)

A small helper script is provided to:
- wait for the service to become ready, and
- guide the evaluator through the UI test, then
- print recently modified files under `logs/` and `reports/`.

Run (in `source_code/`):

```bash
bash ../scripts/minimal_functional_test.sh
```

---

## Troubleshooting / known issues

1) **`OPENAI_API_KEY` missing**
- Symptom: authentication / missing key errors
- Cause: `.env` not created or key not set
- Fix: ensure `source_code/.env` contains `OPENAI_API_KEY=...`, then restart `docker compose up`

2) **Connection reset by peer / service restarts**
- Symptom: `curl: (56) Recv failure: Connection reset by peer` or the UI disconnects
- Likely cause: backend crashed and is restarting (missing env vars, dependency error, runtime exception)
- Fix:
  ```bash
  docker compose logs --tail=200
  docker compose down
  docker compose up --build
  ```

3) **Port `32770` already in use**
- Symptom: Docker cannot bind to `0.0.0.0:32770`
- Cause: another process is using the port
- Fix: stop the conflicting process, or change the host port mapping in `docker-compose.yml`

4) **LLM request failures (rate limit / model not found / network)**
- Symptom: UI shows API errors or timeouts
- Cause: API rate limit, invalid model configuration, or network instability
- Fix: retry, verify `.env` model settings (if present), and confirm internet access

5) **SMT / Z3 errors (local mode)**
- Symptom: import errors if running outside Docker
- Cause: missing `z3-solver` dependency
- Fix: `pip install z3-solver` (Docker path already includes it)

---

## Artifact–paper alignment

This artifact demonstrates the paper’s end-to-end workflow and experiment pipelines.

- **Running example (Minimal Functional Test):**
  - Paper: Section “Agentic Orchestration” → “A Running Example on Income Tax Optimization” (`\label{sec:running-example}`)
  - Figure: “End-to-end income tax optimization” (`\label{fig:income-demo}`)
  - Artifact: Gradio UI workflow with “所得稅” + “下一步” phases

- **RQ1: Constraint Code Synthesis** (`\label{sec:rqs}`): code synthesis / repair pipeline
- **RQ2: Portal-aligned verification** (`\label{sec:rqs}`): author-generated Selenium validation records
- **RQ3: Optimizing Real-World Tax Decisions** (`\label{sec:rq3}`): records and documentation in the archive

### Limitations

For AE Functional, this release prioritizes a minimal end-to-end UI demonstration of the workflow.
Full-scale reproduction of all experimental tables may require longer runtimes and additional compute/API budget.

---

## License

This artifact is released under the **MIT License**. See `LICENSE`.
