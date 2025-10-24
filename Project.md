# Cogito — Comprehensive Project Document (City of Ocala Electric)

**Date:** 2025-10-24  
**Owner:** GMix / Ocala Electric (OT focus)  
**Scope:** This single document combines the **Project Brief** and a full **PROJECT.md** with current state (pre-improvements), plus a detailed specification of the proposed upgrades.

---

## Part A — Project Brief (Executive Summary)

**Cogito** is a self-contained, **local, privacy-first** RAG assistant. It ingests local documents (PDF, DOCX, HTML, CSV/XLSX, TXT) and **images via OCR (Tesseract)**, builds a **vector index (Chroma + `all-MiniLM-L6-v2`)**, and exposes a chat interface that talks to a **local LLM** through **LM Studio** (OpenAI-compatible API). No cloud usage unless explicitly enabled (future toggle).

**Key differentiators**
- **Local-first**: All data & inference on-box; air-gap friendly.
- **Incremental ingestion**: Fingerprint ledger avoids re-embedding unchanged files.
- **Per-Space indexes (planned)**: `data/<space>/` ↔ `chroma_db/<space>/` for clean separation.
- **Domain-aware retrieval (planned)**: Info-density scoring, Survalent/SEL boosts, Troubleshooting mode.
- **Performance modes (planned)**: `fast | balanced | thorough`, warm start to kill first-query lag.
- **Operator UX**: Model picker, depth/temperature sliders, one-click “Check for New / Changed Files.”

**Environment (assumed)**: Windows 11, Python 3.13, LM Studio at `http://localhost:1234`, Tesseract v5.5+, Chroma persistent store.

**Directory layout (Spaces target)**
<repo_root>/
api_server.py
cogito_query.py
cogito_loader.py
data/<space>/
chroma_db/<space>/
.cache/ocr/<space>/
domain_packs/
settings.json

markdown
Copy code
**Legacy fallback:** supports `data/` & `chroma_db/` without `<space>/` as the default Space.

---

## Part B — PROJECT.md (Current State + Detailed Spec)

### 1) What is Cogito? (Overview)
Local-first RAG assistant for Ocala Electric OT. Parses mixed-type docs, builds a vector index, and answers questions grounded in the corpus via a local LLM (LM Studio).

### 2) Current State (Pre-improvements)
- **Vector DB**: Chroma (persisted under `chroma_db/`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: via **LM Studio** @ `http://localhost:1234`
- **Loader (`cogito_loader.py`)**: scans `data/`, fingerprints vs. `db_index.json`, OCRs images, splits text, upserts to Chroma in batches, prints progress & summary.
- **GUI (Streamlit)**: model dropdown, depth/temperature sliders, token cap, chat box (Ctrl+Enter intent), manual refresh, debug timings, button to run incremental loader.
- **Defaults**: local-only; incremental ingestion; 2–5k batch upserts; `.cache/ocr/` ON.
- **Pain points**: cold-start retrieval latency; LLM latency; boilerplate in top-k; mixed vendors; Streamlit keyboard/streaming quirks; single-KB model.

### 3) Requirements
**Must:** local-first; grounded citations; incremental OCR/ingest; practical GUI; Windows-friendly.  
**Should:** lower perceived latency; smarter retrieval; Space isolation.  
**Nice:** domain routing; model management; multi-root import.

### 4) Improvements (Consolidated)
**Performance & UX**
1. SSE streaming + Cancel  
2. Single-pass `fast` mode  
3. Answer cache (Q+params+context hash)  
4. Warm start (DB & prompt cache)  
5. React 3-pane UI with source previews & status bar  
6. Consistent components (buttons, toasts, skeletons)

**Retrieval Quality**
7. Hybrid retrieval (BM25 + Chroma) + RRF  
8. Local reranker (tiny cross-encoder)  
9. Smart chunking v2 (headings/sections)  
10. Info-density + boilerplate suppression  
11. Windowed context (±1 neighbor)  
12. Version/recency bias  
13. Cited-sentence enforcement (+ optional NLI)  
14. Numeric-aware matching

**Data & Ingestion**
15. Spaces (per-KB)  
16. Linked Folders manifest (multi-root with filters + allowlist)  
17. Near-duplicate suppression (SimHash/MinHash)  
18. Table extraction/indexing  
19. Auto-ingest watcher

**Extensibility & Models**
20. Domain Packs  
21. Internal engine (llama.cpp) option  
22. Dual-engine routing (internal + LM Studio: auto/manual/race/consensus)

**Admin & Safety**
23. RBAC & audit (optional)  
24. Export/Playbooks  
25. Eval console

### 5) Detailed Specifications (Implementation-ready)

**5.1 Spaces**
- Paths: `data/<space>/`, `chroma_db/<space>/`, `.cache/ocr/<space>/`, `db_index_<space>.json`
- API: `GET/POST /api/spaces`; all `ask/ingest` take `space`
- UI: header selector + “Create Space” modal
- Acceptance: isolation & citations resolve under the selected Space

**5.2 SSE Streaming + Cancel**
- API: `POST /api/ask_stream` → `text/event-stream` with `data:` tokens & final `[DONE]`
- UI: render tokens, show Cancel (AbortController)
- Acceptance: first token < 1s (warm), cancel < 200ms

**5.3 Fast-mode Single-pass**
- For `mode="fast"`, build one compact context → single LLM call
- Acceptance: ≥30% P50 latency reduction vs. prior fast mode

**5.4 Answer Cache**
- Key: hash(question_norm, mode, depth, model, vendor_scope, troubleshooting, space, top_k_ids)
- Store: on-disk (JSON/SQLite); invalidate on ingest affecting any cached chunk
- Acceptance: hits <100ms

**5.5 Hybrid Retrieval + Reranker**
- BM25 (Whoosh/Elastic-lite) topN≈200 + Chroma topN≈200 → RRF → final topK=depth
- Rerank final 30–60 with `bge-reranker-tiny` (CPU OK)
- Acceptance: +10% precision@k vs. cosine-only; retrieval ≤0.5–0.8s

**5.6 Smart Chunking v2 + Boilerplate**
- Heading/section-aware splitting; store `section_path`
- Regex downrank/strip confidentiality/legal/headers/footers
- Acceptance: boilerplate ≤5% in top-k; better citation usefulness

**5.7 Linked Folders Manifest**
- `spaces.json`:
```json
{
  "default": {
    "roots": [
      {"path":"D:\\\\Vendors\\\\SEL\\\\Manuals","include":["**/*.pdf","**/*.docx"],"exclude":["**/~$*"],"mirror":false,"label":"SEL Docs"},
      {"path":"\\\\\\\\fileserver\\\\Ops\\\\Policies","include":["**/*.pdf","**/*.docx"],"exclude":[],"mirror":true,"label":"Ops Policies"}
    ],
    "allowlist": ["D:\\\\","\\\\\\\\fileserver\\\\Ops\\\\"]
  }
}
Loader scans data/<space>/ + roots (apply filters). If mirror=true, cache extracted text & page thumbnails for robust previews.

Security: enforce allowlist prefixes.

API/UI: list/add/remove roots under Space Settings.

Acceptance: network/local roots supported; previews work offline if mirrored.

5.8 Dual-engine Routing

settings.json engines registry:

json
Copy code
{
  "engines": {
    "internal":{"type":"llamacpp","port":12434,"model_path":"models/gemma-2-9b.Q4_K_M.gguf","ctx":4096,"n_gpu_layers":"auto"},
    "lmstudio":{"type":"openai_proxy","base_url":"http://localhost:1234","model":"google/gemma-3-12b"}
  },
  "engine_policy":"auto",
  "routing":{"fast":"internal","thorough":"lmstudio","max_tokens_threshold":900}
}
Router: by mode/tokens/troubleshooting; “race/consensus” (optional)

UI: engine dropdown; health chips; per-engine timings & cache tags

Acceptance: auto chooses internal for fast asks; failover healthy; race streams the first

5.9 Internal Engine (Optional)

Engine Manager: start/stop/status llama-server.exe (llama.cpp) with tuned ctx & GPU layers

Model Import: add .gguf to models/ with metadata (quant, size, hash)

Acceptance: consistent first-token latency; LM Studio not required when enabled

5.10 Trust & Faithfulness

Prompt requires citation markers per bullet/sentence; optional NLI (DeBERTa-small) to drop non-entailed lines

Acceptance: fewer hallucinations; ≥95% lines cite context

6) API Surface (After Improvements)
bash
Copy code
GET  /api/models
GET  /api/spaces
POST /api/spaces
POST /api/ask
POST /api/ask_stream       # SSE
POST /api/ingest
GET  /api/space_config
POST /api/space_config
GET  /api/preview          # text or PNG
GET  /api/engine/status
POST /api/engine/start
POST /api/engine/stop
7) UI Plan (React + Tailwind)
Three-pane layout; header with Space/Model/Engine selectors + status chips

Chat: streaming, cancel, prompt history (↑/↓), templates, attachments→ingest queue

Sources: citation list with popover preview & “Open page” PNG

Data panel: Check for New/Changed Files + log tail; per-Space stats

Settings: Linked Folders; Domain Pack picker (future)

Hotkeys: Ctrl+Enter (send), Ctrl+K (palette), Ctrl+I (ingest)

8) Milestones
v0.8: Spaces, SSE + cancel, UI shell, answer cache, warm start

v0.9: Hybrid + reranker, smart chunking v2, linked folders

v1.0: Domain Packs, internal engine + dual routing, previews, export, eval console, (optional RBAC & audit)

9) Risks & Mitigations
VRAM with dual engines → choose quant; lazy load; allow CPU fallback

Index bloat → near-duplicate suppression; boilerplate filtering; chunk budgets

Complexity → feature flags; sane defaults; progressive disclosure

Windows quirks → force UTF-8; avoid emoji in console; good errors

10) Acceptance Criteria
P50 total < 8–12s (fast, warm); first token < 1s (SSE)

100% answers cited; P90 page-correct

+10% precision@k vs. cosine baseline (30-query eval)

Incremental ingest without re-embedding unchanged files

Space isolation; robust UX (hotkeys, errors, logs)

11) Quick Start (Current State)
powershell
Copy code
# API
python run_api.py

# UI (Streamlit legacy)
streamlit run cogito_gui.py

# Or UI (React dev)
cd cogito-ui; npm run dev  # open http://localhost:5173

# Ingest (incremental)
$body = @{ space = "default" } | ConvertTo-Json
Invoke-RestMethod -Method POST http://127.0.0.1:8000/api/ingest -ContentType "application/json" -Body $body

# Ask
$body = @{
  question   = "Give me an executive summary of Survalent."
  depth      = 10
  temperature= 0.25
  max_tokens = 900
  model      = "google/gemma-3-12b"
  mode       = "fast"
  vendor_scope = "all"
  troubleshooting = $false
  space      = "default"
} | ConvertTo-Json
Invoke-RestMethod -Method POST http://127.0.0.1:8000/api/ask -ContentType "application/json" -Body $body
Appendix — Glossary
RAG: Retrieval-Augmented Generation (search + LLM)

BM25: Keyword ranking by term frequency/inverse document frequency

RRF: Reciprocal Rank Fusion to combine ranked lists

NLI: Natural Language Inference (entailment checks)

MMR: Maximal Marginal Relevance (diversity in top-k)

yaml
Copy code
---

Want me to also drop a **short one-pager** for executives or a **checklist** version for contractors?
::contentReference[oaicite:0]{index=0}