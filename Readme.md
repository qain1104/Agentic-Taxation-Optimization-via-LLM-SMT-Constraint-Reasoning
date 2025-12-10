# Agentic Taxation Optimization via LLM SMT-Constraint Reasoning — Artifact

This repository contains the artifact for the ICSE 2026 SEIP paper:

> **Agentic Taxation Optimization via LLM SMT-Constraint Reasoning**  
> _ICSE 2026 — Software Engineering in Practice (SEIP)_

The artifact includes the full implementation of the agentic tax optimization system, together with the data and scripts needed to reproduce the main experimental results (RQ1, RQ2, and a planned RQ3).

---

## Badges & Scope

We intend this artifact to qualify for the following ICSE / ACM badges:

- **Artifacts Available**  
  - The full artifact (source code, data, and scripts) will be archived on a long-term repository (e.g., Zenodo) and linked from the camera-ready paper.
- **Artifacts Evaluated — Reusable**  
  - The artifact is documented and structured to support reuse and extension beyond the paper’s experiments, e.g., adapting the pipeline to other tax regimes or constraint problems.

> **Provenance & DOI**  
> - Paper: _Agentic Taxation Optimization via LLM SMT-Constraint Reasoning_ (ICSE 2026 SEIP).  
> - Preprint: `<add arXiv / institutional link here, if any>`  
> - Archived artifact DOI: `<add Zenodo / Figshare DOI here>`  

---

## Repository Layout

At the top level, the repository is organized as follows:

```text
.
├── source_code/      # Full agentic system implementation, Docker setup, SMT code
│   ├── agents/
│   ├── tax_calculators/
│   ├── parsers/
│   ├── reports/
│   ├── logs/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── app.py
│   ├── main.py
│   └── Readme.md     # System-level notes (shipped as-is from the project)
├── rq1/              # Material for RQ1: Constraint Code Synthesis
├── rq2/              # Material for RQ2: Constraint Code Accuracy
├── rq3/              # Material for RQ3: Tax-planning Optimization
├── LICENSE
└── README.md         # Artifact-level README (this file)
