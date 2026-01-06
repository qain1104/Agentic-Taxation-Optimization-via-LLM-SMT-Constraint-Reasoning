python -m taxrag ingest-pdfs --pdf-dir ~/tax_rag/documents --out-dir json_and_csv

python -m taxrag build-chroma --json json_and_csv/all_laws.json --chroma-dir chroma --collection laws_collection

python -m taxrag query --q "..." --k 15 先確認查得到

若要 web search：enable 時 reasoning 改 low