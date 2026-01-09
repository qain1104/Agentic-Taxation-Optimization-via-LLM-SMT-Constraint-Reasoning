# `tax-rag` README

Artifacts related to **RAG-enabled legal retrieval for tax law**:

## Ingesting PDFs

To begin the process of converting tax-related PDF documents into a usable format for the system, use the following command. This will extract and convert the PDFs into JSON and CSV format for further processing.

### Command:

```bash
python -m taxrag ingest-pdfs --pdf-dir ~/tax_rag/documents --out-dir json_and_csv
```

* **`--pdf-dir`**: Directory containing the tax-related PDFs.
* **`--out-dir`**: Directory where the output JSON and CSV files will be stored.

## Building Chroma Vector Store

After ingesting the PDFs, you need to build the Chroma vector store with the parsed data. This allows the system to retrieve relevant legal clauses for code generation.

### Command:

```bash
python -m taxrag build-chroma --json json_and_csv/all_laws.json --chroma-dir chroma --collection laws_collection
```

* **`--json`**: Path to the generated JSON file (from the previous step).
* **`--chroma-dir`**: Directory where the Chroma vector store will be created.
* **`--collection`**: The collection name for storing the legal documents in Chroma.

## Querying Chroma

Once the Chroma vector store is built, you can query it to retrieve the most relevant legal clauses based on a search query.

### Command:

```bash
python -m taxrag query --q "所得" --k 15
```

* **`--q`**: The search query (in this case, "所得" for "income").
* **`--k`**: Number of relevant results to retrieve (default is 15).

## To Enable Web Search

If you want to enable web search during legal reasoning, set the reasoning to **low** when running the pipeline.

```bash
--reasoning low
```
