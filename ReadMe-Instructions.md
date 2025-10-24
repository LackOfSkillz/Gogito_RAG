Cogito — Local, Privacy-First RAG Assistant (City of Ocala Electric)

Status: Active development. **NEW: Streaming answers with real-time token display and intelligent answer caching for instant responses.**

Table of Contents

Overview

Key Features

Architecture

Directory Layout

Prerequisites

Installation

Configuration

Running Cogito

Using the App

Ingestion & OCR

Spaces (Planned) & Data Organization

API Endpoints

Performance Tuning

Security & Privacy

Troubleshooting

Roadmap

Contributing

License

Support

Overview

Cogito is a local, privacy-first retrieval-augmented generation (RAG) assistant tailored for Ocala Electric (OT focus). Cogito ingests on-disk documents (PDF, DOCX, HTML, CSV/XLSX, plain text) and images via OCR (Tesseract), builds a vector index (Chroma + sentence-transformers/all-MiniLM-L6-v2), and answers questions via a local LLM hosted by LM Studio (OpenAI-compatible API).

Local-first by design: No cloud calls unless explicitly added later under an admin toggle.

Key Features

Local document ingestion with incremental updates — only new/changed files are processed.

OCR support for images (Tesseract) with a persistent cache at .cache/ocr/.

Vector search with Chroma (persistent on disk) and MiniLM embeddings.

**Streaming answers with Server-Sent Events (SSE)** — tokens appear as they're generated, first token typically < 1 second (warm).

**Intelligent answer cache** — instant responses for repeat/similar questions with automatic invalidation on content changes.

**Real-time cancel** — stop generation mid-answer with < 200ms response time.

GUI for chat, model selection, depth/temperature sliders, and one-click re-ingestion.

React UI with streaming support, cache statistics, and modern responsive design.

Debug metrics (retrieval time, LLM time, total cycle, cache hit rate).

Windows-friendly: PowerShell helpers, paths, and Unicode-safe logging.

Planned (not yet shipped in this build): Spaces (multi-KB), hybrid retrieval (BM25+vectors), reranker, linked folders, internal engine option, dual-engine routing, previews, Domain Packs, eval console.

Architecture

High level

Ingestion (cogito_loader.py): scan data/, fingerprint, parse, OCR, chunk, upsert to Chroma.

Querying (cogito_query.py): retrieve relevant chunks and call the local LLM via LM Studio.

GUI (cogito_gui.py): Streamlit app to interact with the RAG system (legacy; React UI is being added).

API (api_server.py / run_api.py): FastAPI layer for programmatic access (used by React UI).

Core technologies

Python 3.13, Chroma DB, sentence-transformers, PyMuPDF, pytesseract, bs4/requests, Streamlit, FastAPI, LM Studio.

Directory Layout
C:/Users/gmix/Documents/Cogito_RAG/
├─ cogito_core.py          # constants, schema versioning, hashing, chunking, OCR helpers
├─ cogito_loader.py        # ingestion pipeline (incremental)
├─ cogito_query.py         # retrieval + LLM call via LM Studio
├─ cogito_gui.py           # Streamlit GUI (legacy)
├─ api_server.py           # FastAPI app (used by React UI)
├─ run_api.py              # convenience runner for the API
├─ run_cogito_gui.py       # launcher: dep checks, port checks, opens browser
├─ settings.json           # GUI/API defaults (model, depth, temp, ports)
├─ db_index.json           # fingerprint ledger for incremental ingestion
├─ chroma_db/              # persisted vector store
├─ data/                   # documents to ingest (PDF/DOCX/HTML/CSV/TXT/images)
├─ .cache/
│  └─ ocr/                 # OCR outputs (enabled by default)
└─ requirements.txt


When Spaces land, expect: data/<space>/, chroma_db/<space>/, .cache/ocr/<space>/ and per-space ledgers.

Prerequisites

Windows 11

Python 3.13

Tesseract 5.5+ (installed and on PATH)
Verify: tesseract --version

LM Studio (local server)
Start its server (OpenAI-compatible) on http://localhost:1234, load a chat model (e.g., google/gemma-3-12b)

Optional (for React UI): Node.js 20+, npm, Vite.

Installation

From a PowerShell terminal:

cd C:\Users\gmix\Documents\Cogito_RAG

# (Optional) Create & activate a venv
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# Install Python deps
pip install -U pip wheel
pip install -r requirements.txt


If you see LangChain warnings about langchain_chroma or langchain_openai, you can install their split packages:

pip install -U langchain-chroma langchain-openai

Configuration

Edit settings.json to customize defaults.

{
  "lmstudio_url": "http://localhost:1234",
  "default_model": "google/gemma-3-12b",
  "depth": 10,
  "temperature": 0.3,
  "max_tokens": 900,
  "ui_port": 8501,
  "api_port": 8000,
  "ocr_cache_dir": ".cache/ocr",
  "top_k_cap": 50
}


lmstudio_url: LM Studio server base URL.

default_model: name as reported by GET /v1/models in LM Studio.

depth / temperature / max_tokens: RAG and generation defaults.

ocr_cache_dir: OCR outputs (recommended to keep ON).

ui_port / api_port: ports for Streamlit and API respectively.

Running Cogito
1) Start LM Studio

Open LM Studio

Start the local server (OpenAI-compatible) on http://localhost:1234

Make sure at least one chat model is loaded (e.g., Gemma)

2) Start the API (FastAPI)
cd C:\Users\gmix\Documents\Cogito_RAG
python run_api.py
# Opens: http://127.0.0.1:8000

3) Start the GUI

Streamlit (legacy)

# Either use the launcher:
python run_cogito_gui.py

# Or run Streamlit directly:
streamlit run cogito_gui.py
# Browser: http://localhost:8501


React UI (in development)

# Only if you've set up the React app (cogito-ui):
cd cogito-ui
npm install
npm run dev
# Browser: http://localhost:5173

Using the App

Open the GUI in your browser (http://localhost:8501 for Streamlit or http://localhost:5173 for React dev).

Ensure the model is detected from LM Studio (model dropdown).

Adjust Answer Depth (top-k), Temperature, Max Tokens if needed.

Type a question (Ctrl+Enter to send).

View the answer and citations (file paths and pages).

Use “Check for New / Changed Files” to ingest new docs.

If the first query is slow, subsequent queries should be faster once caches warm up.

Ingestion & OCR

Place documents inside data/ (subfolders OK). Supported types:

PDF, DOCX, HTML (local trees), TXT

CSV/XLSX (parsed to text)

Images (JPG/PNG/TIFF) → OCR via Tesseract

Run the loader manually:

cd C:\Users\gmix\Documents\Cogito_RAG
python cogito_loader.py --progress


What happens

Scanner fingerprints files (path, size, mtime, sha256) against db_index.json

Only new/changed files are parsed

OCR is cached in .cache/ocr/

Text is chunked (token-aware) with metadata (source, page, type)

Chunks upsert to Chroma in bounded batches

A summary prints (docs, chunks, timings)

Delete chroma_db/ to rebuild the vector DB from scratch (rarely needed).

Spaces (Planned) & Data Organization

Current build: single knowledge base under data/ and chroma_db/.
Planned: first-class Spaces for multiple corpora:

data/<space>/       chroma_db/<space>/      .cache/ocr/<space>/


Per-Space ledgers (db_index_<space>.json) and a Space selector in the UI.

API Endpoints

Run python run_api.py and explore http://127.0.0.1:8000.

Common calls (PowerShell examples):

# List models from LM Studio (proxied)
Invoke-RestMethod -Method GET http://127.0.0.1:8000/api/models

# Ask a question (standard)
$body = @{
  question    = "Give me an executive summary of Survalent."
  depth       = 10
  temperature = 0.25
  max_tokens  = 900
  model       = "google/gemma-3-12b"
  mode        = "balanced"
  use_cache   = $true
} | ConvertTo-Json
Invoke-RestMethod -Method POST http://127.0.0.1:8000/api/ask -ContentType "application/json" -Body $body

# Ask with streaming (SSE) - use EventSource or curl for streaming
curl -X POST http://127.0.0.1:8000/api/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain SCADA architecture","depth":10,"mode":"balanced"}' \
  -N

# Get cache statistics
Invoke-RestMethod -Method GET http://127.0.0.1:8000/api/cache/stats

# Clear cache
Invoke-RestMethod -Method POST http://127.0.0.1:8000/api/cache/clear

# Trigger ingest (incremental)
$body = @{ } | ConvertTo-Json
Invoke-RestMethod -Method POST http://127.0.0.1:8000/api/ingest -ContentType "application/json" -Body $body


### NEW Streaming Endpoint

**POST /api/ask_stream** returns Server-Sent Events (SSE) with real-time token generation:

```
data: {"type":"metadata","sources":[...],"mode":"balanced"}

data: {"type":"token","content":"The"}
data: {"type":"token","content":" Survalent"}
data: {"type":"token","content":"ONE"}
...
data: {"type":"done","timings":{...}}

data: [DONE]
```

- **First token latency**: Typically < 1 second on warm API
- **Cancellation**: Client abort stops generation within 200ms
- **Use case**: React UI uses EventSource API with AbortController

### Cache Endpoints

**GET /api/cache/stats** - Returns cache metrics:
```json
{
  "total_entries": 45,
  "total_hits": 123,
  "hit_rate": 0.732,
  "oldest_entry_age_s": 86400,
  "newest_entry_age_s": 120,
  "size_mb": 2.3
}
```

**POST /api/cache/clear** - Clears entire cache (use after major ingestion changes)

Additional endpoints may include Spaces and Previews in future versions.

Performance Tuning

LLM choice & quantization (LM Studio): Prefer faster quants (e.g., Q4_K_M). Reduce context length if not needed (2–4k).

Tokens: Lower max_tokens (e.g., 600–900) when possible.

Depth: Start with depth 8–12; higher depth can slow retrieval/generation.

OCR cache: Keep .cache/ocr/ enabled.

**Answer cache**: Enabled by default. Provides sub-100ms responses for repeat queries. Automatically invalidated when source chunks change.

**Streaming**: Use `/api/ask_stream` for better perceived performance — first token appears in < 1s (warm), users can cancel long answers.

Batch sizes: Loader uses bounded upsert batches to avoid driver errors.

Warm start: Keep API running to avoid first-query lag.

Cache Statistics: Monitor via `/api/cache/stats` — target hit rate > 30% for typical ops workloads.

Security & Privacy

Local-only by default. No cloud calls unless explicitly enabled in future.

All documents, indexes, and caches live on disk in the project directory.

For sensitive environments, isolate the machine/network and restrict access to the project folder.

Troubleshooting

Unicode/emoji errors in console (Windows)

Run Python with UTF-8 and avoid emojis in console prints. We’ve minimized emojis in logs.

Ensure PowerShell uses UTF-8: chcp 65001 (if needed).

Empty answers

Ensure LM Studio has an active model and the model name matches.

Increase depth slightly (e.g., 12–14) or refine your query.

Verify relevant documents were ingested (check loader summary).

Slow first query

Expected while the DB and model warm up. Subsequent queries are faster.

Keep the API process running.

Re-ingestion didn’t pick up files

Check that files were placed under data/ and not excluded.

Verify db_index.json timestamp and run with --progress for details.

Roadmap

**v0.8 (COMPLETED)**: ✅ Streaming answers with SSE + Cancel, ✅ Answer Cache with automatic invalidation

v0.9: Spaces (multi-KB collections), React UI production polish, hybrid retrieval (BM25+vectors + RRF), local reranker, smart chunking v2, linked folders

v1.0: Domain Packs, internal engine (llama.cpp) option, dual-engine routing, page previews, exports, eval console, (optional RBAC & audit)

Contributing

Use feature branches; open PRs with clear descriptions and reproduction steps.

Keep Windows paths in mind; prefer PowerShell snippets.

Add unit tests or at least end-to-end scripts for loader and query flows.