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
- **[Testing]** Provided a **Minimal Functional Test** workflow intended to complete within **≤10 minutes (excluding Docker build time)**.
- **[Environment]** Documented dependency configuration (Docker-first), Python base image version, and SMT solver usage.
- **[Transparency]** Documented expected outputs (incl. a deterministic `ae_outputs/` folder) and added a troubleshooting section for common LLM API and runtime issues.
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

The system will enter the Phase3 : Final Confirmation Before Execution

Then type and send:

> 下一步

Wait for the system to generate an optimization result and a report.

**UI success criteria:**  
- a generated result/report is displayed in the UI (Markdown-like text is acceptable)

---

## Expected outputs (what to look for)

This artifact is UI-driven and may produce different filenames across configurations.  
To avoid guesswork, **the Minimal Functional Test defines deterministic outputs under `ae_outputs/`**.

### Outputs produced by the Minimal Functional Test helper

Run (from the artifact root; see below):

```bash
bash scripts/minimal_functional_test.sh
```

This creates (or updates) a folder `ae_outputs/` in the artifact root:

```text
ae_outputs/
  ├── openapi.json
  ├── docker_logs_tail.txt
  ├── recent_files.txt
  └── common_outputs.txt
```

**Meaning of each file:**
- `openapi.json`: captured OpenAPI spec, proving the service is reachable and responding.
- `docker_logs_tail.txt`: last 200 lines of service logs for debugging and verification.
- `recent_files.txt`: files under `source_code/` modified in the last ~10 minutes after the UI run.
- `common_outputs.txt`: a best-effort listing of files under common output folders (logs/reports/outputs).

**Functional success criteria (outputs):**
1) `ae_outputs/openapi.json` exists and is non-empty; and  
2) `ae_outputs/docker_logs_tail.txt` contains recent runtime logs (timestamped lines from uvicorn/gradio); and  
3) At least one of the following holds:
   - `ae_outputs/recent_files.txt` lists files under `source_code/logs/` updated during the run, **or**
   - the UI shows the final report/result in Phase 3 (manual check).

### Typical application output locations (if enabled in your configuration)

When file logging/report persistence is enabled, you should additionally see outputs under:

- **Logs:** `source_code/logs/` (e.g., `*.log`)
- **Reports / artifacts:** `source_code/reports/` (e.g., `*.md`, `*.json`)

If your deployment does not persist reports to disk, the UI display + `ae_outputs/` artifacts remain the authoritative success signal for AE Functional.

---

## Execution time & resource expectations

Measured/estimated for a typical laptop/desktop with Docker installed (no GPU required).

### Typical runtime (approx.)

| Step | Typical time | Notes |
|---|---:|---|
| `docker compose build` (first time) | ~10 min | dependency download + image build |
| `docker compose up` → ready | ~30–90 sec | until `/openapi.json` responds |
| Minimal Functional Test (UI run) | ~2–8 min | depends mainly on LLM latency/rate limits |
| `bash scripts/minimal_functional_test.sh` | ≤10 min | excludes Docker build time |

> If you want to record your machine’s exact numbers, run:
> - `time docker compose build`
> - `time curl -sSf http://localhost:32770/openapi.json >/dev/null`

### Resource usage (approx.)

- **CPU:** 1–2 cores during normal serving; can spike during startup/build
- **RAM:** 2–4 GB typical (8 GB recommended for comfort)
- **Disk:** 3–6 GB (Docker image layers + temporary build cache)

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
10. In the artifact root (one level above `source_code/`), run `bash scripts/minimal_functional_test.sh`  
11. Inspect `ae_outputs/` and (optionally) `source_code/logs/`, `source_code/reports/`

---

## Minimal Functional Test suite (scripted helper)

Run this from the **artifact root** (the folder that contains `source_code/` and `scripts/`):

```bash
bash scripts/minimal_functional_test.sh
```

It will:
- wait for the service liveness endpoint (`/openapi.json`),
- guide you through the UI income-tax example,
- and save AE-verification artifacts under `ae_outputs/`.

---

## Troubleshooting / known issues

1) **`OPENAI_API_KEY` missing**
- Symptom: authentication / missing key errors
- Cause: `.env` not created or key not set
- Fix: ensure `source_code/.env` contains `OPENAI_API_KEY=...`, then restart `docker compose up`

2) **External Docker network missing (older versions)**
- Symptom: `network fin-network declared as external, but could not be found`
- Cause: an older `docker-compose.yml` referenced an external network not present on the host
- Fix (old versions only): `docker network create fin-network`, then re-run compose  
  *(Current Zenodo version removes this external network requirement.)*

3) **Connection reset by peer / service restarts**
- Symptom: `curl: (56) Recv failure: Connection reset by peer` or the UI disconnects
- Likely cause: backend crashed and is restarting (missing env vars, dependency error, runtime exception)
- Fix:
  ```bash
  cd source_code
  docker compose logs --tail=200
  docker compose down
  docker compose up --build
  ```

4) **Port `32770` already in use**
- Symptom: Docker cannot bind to `0.0.0.0:32770`
- Cause: another process is using the port
- Fix: stop the conflicting process, or change the host port mapping in `docker-compose.yml`

5) **LLM request failures (rate limit / model not found / network)**
- Symptom: UI shows API errors or timeouts
- Cause: API rate limit, invalid model configuration, or network instability
- Fix: retry, verify `.env` model settings (if present), and confirm internet access

6) **SMT / Z3 errors (local mode)**
- Symptom: import errors if running outside Docker
- Cause: missing `z3-solver` dependency
- Fix: `pip install z3-solver` (Docker path already includes it)

---

## Artifact–paper alignment

This artifact directly corresponds to the workflow and evaluation in the paper:

- **End-to-end workflow overview (Paper Figure 1):**  
  The paper’s full pipeline—LLM-assisted constraint-code synthesis + portal validation (upper panel), then agentic optimization using verified Z3 code (lower panel)—is implemented in the artifact’s `source_code/` service.

- **Running example (Paper Section 4.3, Figure 3):**  
  The Minimal Functional Test follows the same interaction pattern as the paper’s income-tax running example: select a tax module → fill missing fields → specify (optional) constraints/free variables → generate an auditable optimization report.  
  This is exactly what the Gradio UI demonstrates in Step 0 + Phase 1–3 in this README.

- **RQ1 — Constraint Code Synthesis (Paper Section 5.4):**  
  Implementation and records for the constraint synthesis/repair workflow are included under:
  - `source_code/code_synthesis/` (codegen + repair pipeline; detailed docs inside that folder)

- **RQ2 — Portal-aligned verification (Paper Section 5.5):**  
  Author-generated Selenium oracle validation records and reproduction notes are included under:
  - `source_code/selenium_test_records/` (Selenium records/logs and scripts)

- **RQ3 — Real-world optimization tasks (Paper Section 5.6):**  
  Task records and supporting material are included under:
  - `RQ3/`

### Limitations (what is and isn’t reproduced for AE Functional)

- **External dependency:** The functional pipeline requires internet access and a valid `OPENAI_API_KEY` (no offline mode in this release).
- **Scope for AE Functional:** The Minimal Functional Test demonstrates the end-to-end workflow on one representative scenario (income tax UI run).  
  Reproducing all RQ1/RQ2/RQ3 tables end-to-end is possible with the included materials, but may require additional runtime and LLM/API budget beyond the quick AE Functional check.
- **Selenium/portal verification:** Full RQ2 reproduction may require a browser runtime and stable access to the MoF portal, which can be sensitive to network conditions and portal updates.

---

## License

This artifact is released under the **MIT License**. See `LICENSE`.
