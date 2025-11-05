# Wikipedia RAG Question Answering System

A Retrieval-Augmented Generation (RAG) system built **entirely with LangChain**, served by **FastAPI** and a **Streamlit** UI.

**Pipeline**

-   Ingest Wikipedia pages by **title** via LangChain `WikipediaLoader`, then **split** with `RecursiveCharacterTextSplitter`.
    
-   Build a **FAISS** vector store using **Hugging Face** embeddings (`langchain-huggingface`).
    
-   **Retrieve (FAISS) → Cross-Encoder rerank (sentence-transformers) → LLM generate** (local HF or OpenAI) with inline citations `[[n]]`.
    
-   Serve `/ask` via FastAPI; the Streamlit UI calls the same API.
    

----------

## Requirements

-   **Python 3.11–3.12 recommended** (Py3.13 may have fewer prebuilt wheels for some deps on Windows).
    
-   Windows PowerShell (commands below), or Bash (Mac/Linux).
    
-   Optional: OpenAI API key (if using OpenAI provider).
    

----------

## Quickstart (Windows PowerShell)

### 0) Create venv and install

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

```

### 1) Ingest a few Wikipedia titles

Edit `data\seed_titles.txt` (one title per line), then:

```powershell
python -m src.lc_cli ingest --titles data\seed_titles.txt --out data\chunks.jsonl --lang en

```

### 2) Build the LangChain FAISS store

```powershell
python -m src.lc_cli build --chunks data\chunks.jsonl --vs_dir artifacts\lc_faiss `
  --embed_model sentence-transformers/all-MiniLM-L6-v2

```

### 3) Run the API (choose ONE provider)

**A) Local Hugging Face (offline)**

```powershell
python -m pip install -U accelerate safetensors
$env:PROVIDER = "hf"
$env:HF_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"   # or "SmolLM2-360M"
$env:VS_DIR   = "artifacts\lc_faiss"
$env:MAX_TOKENS = "128"; $env:TEMPERATURE = "0.0"
python -m uvicorn src.lc_api:app --host 127.0.0.1 --port 8000 --reload

```

**B) OpenAI (requires quota)**

```powershell
$env:PROVIDER = "openai"
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_MODEL = "gpt-4o-mini"
$env:VS_DIR = "artifacts\lc_faiss"
python -m uvicorn src.lc_api:app --host 127.0.0.1 --port 8000 --reload

```

### 4) Run the Streamlit UI

```powershell
# new terminal, same venv
$env:API_URL = "http://127.0.0.1:8000"
python -m streamlit run ui\app.py

```

The UI lets you set Top-K and the rerank pool, calls `/ask`, and displays the answer with sources.

----------

## CLI (ingest / build / ask)

```powershell
# Ingest Wikipedia pages -> chunks.jsonl
python -m src.lc_cli ingest --titles data\seed_titles.txt --out data\chunks.jsonl --lang en
# Options: --chunk_size 1000 --chunk_overlap 200

# Build FAISS from chunks
python -m src.lc_cli build --chunks data\chunks.jsonl --vs_dir artifacts\lc_faiss `
  --embed_model sentence-transformers/all-MiniLM-L6-v2

# Ask a question via the RAG pipeline
python -m src.lc_cli ask "Who founded SpaceX?" --vs_dir artifacts\lc_faiss `
  --k 8 --rerank_top 20 --provider hf --hf_model HuggingFaceTB/SmolLM2-360M-Instruct `
  --max_tokens 128 --temperature 0.0
# (OpenAI: --provider openai --openai_model gpt-4o-mini with OPENAI_API_KEY set)

```

**Flow:** `ingest` (create chunks) → `build` (create index) → `ask` (query the index).

----------

## Project Structure

```
wikipedia-rag-langchain-streamlit/
├── data/
│   ├── seed_titles.txt
│   └── sample_qa.jsonl
├── artifacts/                   # created at runtime
├── src/
│   ├── lc_ingest.py            # Wikipedia → chunks.jsonl
│   ├── lc_build.py             # chunks.jsonl → FAISS vector store
│   ├── lc_rerank.py            # Cross-Encoder reranker (sentence-transformers)
│   ├── lc_chain.py             # retrieve → rerank → LLM with citations
│   ├── lc_cli.py               # CLI: ingest/build/ask
│   └── lc_api.py               # FastAPI server (/ask)
├── ui/
│   └── app.py                  # Streamlit UI calling the API
├── requirements.txt
└── README.md

```

----------

## Notes

-   **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (fast). For higher recall, try `BAAI/bge-base-en-v1.5`.
    
-   **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
    
-   **Citations:** inline `[[n]]` map to the contexts shown in the UI.
    
-   **CPU tips:** keep `MAX_TOKENS` modest (e.g., 64–160) and consider `repetition_penalty=1.1–1.2` for cleaner generations.
    


----------