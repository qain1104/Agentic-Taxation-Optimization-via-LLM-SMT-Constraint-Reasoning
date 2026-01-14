#!/usr/bin/env bash
set -euo pipefail

# taxrag_pipeline.sh
# Usage examples:
#   bash taxrag_pipeline.sh ingest   --pdf-dir ~/tax_rag/documents --out-dir json_and_csv
#   bash taxrag_pipeline.sh build    --out-dir json_and_csv --chroma-dir chroma --collection laws_collection
#   bash taxrag_pipeline.sh query    --q "所得" --k 15 --chroma-dir chroma --collection laws_collection
#   bash taxrag_pipeline.sh all      --pdf-dir ~/tax_rag/documents --out-dir json_and_csv --chroma-dir chroma --collection laws_collection --q "所得" --k 15

CMD="${1:-all}"
shift || true

# defaults (can be overridden by flags)
PDF_DIR="${HOME}/tax_rag/documents"
OUT_DIR="json_and_csv"
CHROMA_DIR="chroma"
COLLECTION="laws_collection"
QUERY="所得"
K=15

usage() {
  cat <<'EOF'
taxrag_pipeline.sh

Subcommands:
  ingest   Convert PDFs -> JSON/CSV
  build    Build Chroma vector store from JSON
  query    Query Chroma
  all      Run ingest + build + query

Common flags:
  --pdf-dir <dir>        (default: ~/tax_rag/documents)
  --out-dir <dir>        (default: json_and_csv)
  --chroma-dir <dir>     (default: chroma)
  --collection <name>    (default: laws_collection)

Query flags:
  --q <query>            (default: 所得)
  --k <int>              (default: 15)

Examples:
  bash taxrag_pipeline.sh ingest --pdf-dir ~/tax_rag/documents --out-dir json_and_csv
  bash taxrag_pipeline.sh build  --out-dir json_and_csv --chroma-dir chroma --collection laws_collection
  bash taxrag_pipeline.sh query  --q "所得" --k 15 --chroma-dir chroma --collection laws_collection
  bash taxrag_pipeline.sh all    --pdf-dir ~/tax_rag/documents --out-dir json_and_csv --chroma-dir chroma --collection laws_collection --q "所得" --k 15
EOF
}

# parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdf-dir)      PDF_DIR="$2"; shift 2;;
    --out-dir)      OUT_DIR="$2"; shift 2;;
    --chroma-dir)   CHROMA_DIR="$2"; shift 2;;
    --collection)   COLLECTION="$2"; shift 2;;
    --q)            QUERY="$2"; shift 2;;
    --k)            K="$2"; shift 2;;
    -h|--help)      usage; exit 0;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

run_ingest() {
  echo "=================================================="
  echo "[taxrag] ingest-pdfs"
  echo "  pdf-dir : ${PDF_DIR}"
  echo "  out-dir : ${OUT_DIR}"
  echo "=================================================="
  python -m taxrag ingest-pdfs --pdf-dir "${PDF_DIR}" --out-dir "${OUT_DIR}"
}

run_build() {
  local JSON_PATH="${OUT_DIR}/all_laws.json"
  if [[ ! -f "${JSON_PATH}" ]]; then
    echo "[ERROR] Missing JSON: ${JSON_PATH} (run ingest first?)" >&2
    exit 1
  fi
  echo "=================================================="
  echo "[taxrag] build-chroma"
  echo "  json      : ${JSON_PATH}"
  echo "  chroma-dir: ${CHROMA_DIR}"
  echo "  collection: ${COLLECTION}"
  echo "=================================================="
  python -m taxrag build-chroma --json "${JSON_PATH}" --chroma-dir "${CHROMA_DIR}" --collection "${COLLECTION}"
}

run_query() {
  echo "=================================================="
  echo "[taxrag] query"
  echo "  q         : ${QUERY}"
  echo "  k         : ${K}"
  echo "  chroma-dir: ${CHROMA_DIR}"
  echo "  collection: ${COLLECTION}"
  echo "=================================================="
  # If your taxrag query command supports chroma-dir/collection flags, keep them.
  # If not, remove those two flags.
  python -m taxrag query --q "${QUERY}" --k "${K}" --chroma-dir "${CHROMA_DIR}" --collection "${COLLECTION}"
}

case "${CMD}" in
  ingest) run_ingest;;
  build)  run_build;;
  query)  run_query;;
  all)
    run_ingest
    run_build
    run_query
    ;;
  -h|--help) usage;;
  *)
    echo "[ERROR] Unknown subcommand: ${CMD}" >&2
    usage
    exit 1
    ;;
esac

echo "[DONE] taxrag_pipeline.sh ${CMD}"
