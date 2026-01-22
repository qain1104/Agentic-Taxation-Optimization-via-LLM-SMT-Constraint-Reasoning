#!/usr/bin/env bash
set -euo pipefail

BASE_CMD="python source_code/code_synthesis/tax_agent_pipeline.py"

INPUT_DIR="source_code/code_synthesis/inputs"
SAMPLES_DIR="source_code/code_synthesis/protal_samples"
CHROMA_DIR="source_code/code_synthesis/chroma"
COLLECTION="laws_collection"
EXTRA_REF="source_code/code_synthesis/refs/114_numbers.txt"

OUT_DIR="source_code/code_synthesis/outputs"
mkdir -p "${OUT_DIR}"

INPUT_FILES=(
  "income_tax.txt"
  "business_income.txt"
  "business_tax.txt"
  "cargo_tax.txt"
  "estate_input.txt"
  "foreign_income_tax.txt"
  "gift_tax_input.txt"
  "security_futures.txt"
  "special_goods_services.txt"
  "ta_tax.txt"
)

for INPUT_FILE in "${INPUT_FILES[@]}"; do
  INPUT_PATH="${INPUT_DIR}/${INPUT_FILE}"

  # input 檔名 -> schema + samples 檔名（注意：_input 的 samples 檔名沒有 _input）
  case "${INPUT_FILE}" in
    income_tax.txt)            SCHEMA="income_tax";             SAMPLE_STEM="income_tax" ;;
    business_income.txt)       SCHEMA="business_income";        SAMPLE_STEM="business_income" ;;
    business_tax.txt)          SCHEMA="business_tax";           SAMPLE_STEM="business_tax" ;;
    cargo_tax.txt)             SCHEMA="cargo_tax";              SAMPLE_STEM="cargo_tax" ;;
    estate_input.txt)          SCHEMA="estate";                 SAMPLE_STEM="estate" ;;
    foreign_income_tax.txt)    SCHEMA="foreign_income_tax";     SAMPLE_STEM="foreign_income_tax" ;;
    gift_tax_input.txt)        SCHEMA="gift_tax";               SAMPLE_STEM="gift_tax" ;;
    security_futures.txt)      SCHEMA="security_futures";       SAMPLE_STEM="security_futures" ;;
    special_goods_services.txt)SCHEMA="special_goods_services"; SAMPLE_STEM="special_goods_services" ;;
    ta_tax.txt)                SCHEMA="ta_tax";                 SAMPLE_STEM="ta_tax" ;;
    *)
      echo "[ERROR] Unknown INPUT_FILE: ${INPUT_FILE}"
      exit 1
      ;;
  esac

  SAMPLES_PATH="${SAMPLES_DIR}/${SAMPLE_STEM}_samples.json"
  OUT_PATH="${OUT_DIR}/${SAMPLE_STEM}_solver.py"

  if [[ ! -f "${INPUT_PATH}" ]]; then
    echo "[ERROR] Missing input file: ${INPUT_PATH}"
    exit 1
  fi

  if [[ ! -f "${SAMPLES_PATH}" ]]; then
    echo "[ERROR] Missing samples file for ${INPUT_FILE}: ${SAMPLES_PATH}"
    exit 1
  fi

  echo "=================================================="
  echo "Running pipeline"
  echo "  schema  : ${SCHEMA}"
  echo "  input   : ${INPUT_PATH}"
  echo "  samples : ${SAMPLES_PATH}"
  echo "  out     : ${OUT_PATH}"
  echo "=================================================="

  ${BASE_CMD} \
    --input "${INPUT_PATH}" \
    --samples "${SAMPLES_PATH}" \
    --chroma-dir "${CHROMA_DIR}" \
    --collection "${COLLECTION}" \
    --schema "${SCHEMA}" \
    --extra-ref "${EXTRA_REF}" \
    --reasoning-codegen high \
    --reasoning-repair high \
    --max-attempts 5 \
    --out "${OUT_PATH}" \
    --show-llm-notes

  echo ""
done

echo "All pipelines finished."
