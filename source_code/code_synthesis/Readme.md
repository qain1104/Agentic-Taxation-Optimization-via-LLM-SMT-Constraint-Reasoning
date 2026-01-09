Artifacts related to **code synthesis layer for tax law generation**:

## Running the Tax Agent Pipeline

The Tax Agent Pipeline combines RAG-enabled legal retrieval with high-reasoning code generation for tax law-related applications. To start the pipeline, execute the following command from the project root directory.

### Command:

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
```

* **`--input`**: Path to the input text file containing the tax law scenario (e.g., `income_tax.txt`).
* **`--samples`**: Path to the samples JSON file containing the test scenarios.
* **`--chroma-dir`**: Path to the directory where the Chroma vector store is located.
* **`--collection`**: The collection name for the legal documents stored in Chroma (e.g., `laws_collection`).
* **`--schema`**: Schema to apply to the input data (e.g., `income_tax`).
* **`--extra-ref`**: Additional reference file that provides extra context for the pipeline (e.g., `114_numbers.txt`).
* **`--reasoning-codegen`**: Level of reasoning applied to code generation (`high` for detailed reasoning).
* **`--reasoning-repair`**: Level of reasoning applied to code repair (`high` for more robust repairs).
* **`--show-llm-notes`**: Option to show notes from the language model during code generation.

## Pipeline Overview

This pipeline uses **Legal Clause Retrieval** from the Chroma vector store and synthesizes code based on those legal clauses using a high-reasoning code generation model (e.g., GPT). It integrates tax laws into code logic and also performs repair when necessary for correct tax computation.


## quick start:

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