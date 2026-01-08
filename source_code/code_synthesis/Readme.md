## Run the Tax Agent Pipeline

To start the tax agent pipeline with RAG-enabled legal retrieval and
high-reasoning code synthesis, run the following command from the
project root directory:

```bash
python source_code/code_synthesis/tax_agent_pipeline.py \
  --input source_code/code_synthesis/inputs/income_tax.txt \
  --samples source_code/code_synthesis/protal_samples/income_tax_samples.json \
  --chroma-dir source_code/code_synthesis/chroma \
  --collection laws_collection \
  --schema income_tax \
  --extra-ref source_code/code_synthesis/refs/114_numbers.txt \
  --reasoning-codegen high \
  --reasoning-repair high \
  --show-llm-notes