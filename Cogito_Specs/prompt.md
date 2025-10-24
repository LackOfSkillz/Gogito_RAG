Got it — here’s a **single, copy-pasteable SpecKit / Specify design prompt** that captures our **latest Cogito** requirements and context. Drop this in as your “project brief” to generate a full spec (PRD + architecture + APIs + test plan).

---

# **SpecKit Prompt — Cogito (City of Ocala Electric OT AI Assistant)**

**Project Name:** Cogito — City of Ocala Electric OT AI Assistant
**Current Version:** v0.7-dev (pre-v1)
**Goal:** A local, on-prem RAG assistant for SCADA/OT/networking knowledge that runs against a private document corpus; secure, fast, easy to operate by non-ML users; multi-department “Brains” in the roadmap.

---

## 1) Context & Constraints

* **Environment:** Primarily Windows 10/11 workstations; some Linux servers later.
* **Air-gapped / privacy sensitive:** All core tasks must work fully offline.
* **Local LLM:** LM Studio running a Gemma-3 family model via local server API.
* **Tech Stack (current dev):**

  * **Frontend:** Streamlit (Python) GUI
  * **RAG Orchestration:** LangChain
  * **Vector DB:** Chroma (SQLite persistence); schema versioning maintained in JSON
  * **Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (HuggingFace)
  * **OCR:** Tesseract (Windows installer available) via `pytesseract`
  * **HTML parsing:** `readability-lxml` + `BeautifulSoup4` (no Playwright needed unless site requires JS)
* **LM Studio:** Auto-discover models via `/v1/models`; user selects active model in GUI; default port 1234.
* **Ports:** Streamlit 8501 by default (configurable).
* **No cloud dependencies** required for core. Optional web fallback is opt-in.

---

## 2) Personas

* **SCADA Operator:** Needs quick answers from Survalent manuals, alarm behavior, polling settings, procedures.
* **Network Engineer:** Troubleshoots switches/routers/firewalls; wants config parsing, topology hints.
* **Helpdesk/IT Generalist:** Searches KB/FAQs, triages common calls.
* **Department Leads (Finance, Legal, Maintenance):** Separate “Brains” with access controls (roadmap).
* **Admin:** Manages data ingestion, deletions, model selection, logging, access policies.

---

## 3) High-Level Objectives

1. Fast, relevant answers grounded in local content (PDFs/HTML/Office docs/images with OCR).
2. Clear UX: search, answer depth slider, temperature, progress bars, visible loading states.
3. Incremental ingestion with **fingerprinting** (hashing) to avoid reprocessing unchanged files.
4. Basic **delete/restore** capabilities for documents/folders from GUI.
5. Extensible architecture for multi-user/multi-brain and pgvector later.

---

## 4) Scope — **MVP+ (v0.7 line) Requirements**

### 4.1 Ingestion (“Build Knowledge Base”)

* **Recursive** scan of `/data` (any depth), directory-agnostic on first run (auto-create structure).
* **Supported types (initial):** `.pdf, .docx, .pptx, .xlsx, .csv, .txt, .md, .html, .htm, .jpg, .jpeg, .png`
* **OCR:** Run Tesseract on images; associate text to image metadata.
* **HTML:** Parse with readability + bs4; strip boilerplate; store canonical URL (if known).
* **Chunking:**

  * Semantic text chunking with stable IDs (include file hash + page/frame indices).
  * Default chunk size ~750 tokens, overlap ~100 (configurable), doc-type aware (tables/slides shorter).
* **Metadata:** `source_path, file_hash, file_type, page_or_slide, created_at, modified_at, brain_name, tags[]`
* **Fingerprinting:** SHA-256 per file; skip unchanged; detect deletes and remove from DB.
* **Batching:** Safe batch size for Chroma; streaming add with backpressure.
* **Parallelism:** Multithreaded parsing; **single-writer** DB commit (file-lock to avoid corruption).
* **Logging & Profiling:** Structured logs (JSON lines) with timings per phase; rotating logs.
* **Schema Versioning:** `schema.json` stored; loader refuses to write if version mismatch (offers migration).

### 4.2 Retrieval & Answering

* **Retriever:** Hybrid keyword + vector search (if feasible); otherwise vector search with metadata filters.
* **Depth slider (k):** 1–8 (UI control) to adjust number of retrieved chunks.
* **Temperature slider:** 0.0–1.5.
* **Model autodiscovery:** Query LM Studio `/v1/models`; allow model selection in sidebar.
* **Answer streaming:** Show tokens as they arrive; show “Retrieval/LLM/Total” times.
* **Citations:** Return source snippets with file name/page and linkable path.
* **If no good answer:** Return graceful “I don’t know from local KB; (optional) web search?” prompt.
* **(Optional toggle)**: Web search fallback (if user explicitly enables; logs marked as external).

### 4.3 GUI (Streamlit) Features

* **Splash/Loading state:** On startup, show “Cogito is loading…” with spinner; initial cache warm.
* **Compact controls:** Small sliders; **Submit** button directly under input; **Ctrl+Enter** submits.
* **Sidebar panels:**

  * **Chat Assistant** (default)
  * **Navigation/DB search** (shows presence of topics; simple keyword + metadata filters)
  * **Database Info** (counts by type, chunks, last ingested, brains)
  * **Manage Data** (upload files/folders, delete from DB, re-ingest selection)
  * **Settings** (model picker, depth, temperature, top-k rerank, OCR toggle)
* **Progress bars:** For ingestion and OCR; per-phase percent & ETA (coarse).
* **Debug panel (collapsible):** Shows last timings, batch sizes, and recent errors.

### 4.4 “Brains” (Namespaces) — MVP behavior

* **Brain concept:** A namespace label attached to each document (e.g., `survalent`, `networking`, `finance`).
* **User can select one or multiple brains** as retrieval filter(s).
* **No auth gating in MVP** (single-user dev mode); multi-user/ACL is roadmap.

### 4.5 Delete / Remove

* From GUI:

  * Delete a **document** (uses file hash/path to remove chunks).
  * Delete a **folder** (recursively remove all belonging docs).
  * Persist a tombstone so deleted items don’t immediately re-ingest unless restored.
* Confirm dialogs and undo/restore option if source file still exists.

---

## 5) Roadmap (Post-MVP / v0.9 → v1.0)

* **Multi-user & ACL:** Users + roles (Admin, Department, Read-only), SSO optional, row-level filtering per brain.
* **Switchable Vector Backend:** Abstract to `vector_backend.py`; add **pgvector** (PostgreSQL) option.
* **Advanced reranking:** Re-rank retrieved chunks with cross-encoder or local reranker model.
* **Knowledge graphs / structured extraction:** Entities & relationships for OT assets, alarms, feeders, etc.
* **Network config parsing module:** Recognize vendor syntaxes (Cisco/Juniper/Aruba/etc.), build topology hints.
* **Schema migration tool:** Upgrade existing DB across versions safely (backups, diffs).
* **Audit & observability:** Query logs, retention policy, export.
* **Packaging:** `pyinstaller` build → `cogito_setup.exe` and a separate `cogito_gui.exe` runner.
* **Helpdesk IVR integration:** Optional voice flow: ASR → RAG → TTS; escalate to human if confidence low.

---

## 6) Non-Functional Requirements

* **Performance targets (dev):**

  * Retrieval budget target: ≤ 5–10 s on 30k–60k chunks with warm cache.
  * Answer total time: ≤ 15 s median on typical questions (with local Gemma-3 12B).
* **Reliability:**

  * File-lock to enforce single ingestion writer.
  * Safe shutdown and resume of ingestion; idempotent re-runs.
* **Security & Privacy:**

  * No data leaves machine unless user enables web fallback.
  * Logs redact PII where feasible; store only minimal request metadata.
* **Operability:**

  * One-click GUI launcher; port conflicts handled; status banners.
  * Clear error messages + remediation hints.

---

## 7) Data Model (Core Entities)

* **FileRecord**: `{file_path, file_hash, file_type, brain, size, created_at, modified_at, status}`
* **DocChunk**: `{chunk_id, file_hash, page_index, text, tokens, embedding_vector_ref, metadata{…}}`
* **IndexState**: `{schema_version, embeddings_model_id, vector_backend, created_at, last_ingest_at}`
* **Tombstone** (optional): `{file_hash, deleted_at, reason}`

---

## 8) APIs (Internal Modules)

* **Ingestion API:**

  * `scan_files(root) -> List[FileRecord]`
  * `diff_fingerprints(new_files) -> {new, modified, deleted}`
  * `parse(file) -> List[DocChunkInput]` (type-specific handlers; OCR & HTML)
  * `embed(chunks, batch_size) -> embeddings`
  * `persist(chunks, embeddings, metadata)`
  * `delete_by_file_hash(hash)` / `delete_by_folder(path)`
* **Query API:**

  * `retrieve(query, brains[], k, filters) -> List[DocChunk]`
  * `generate_answer(query, chunks, model, temperature) -> stream|text`
  * `metrics() -> {retrieval_time, llm_time, total, num_chunks}`
* **Admin API (GUI hooks):**

  * `list_docs(filters)` / `list_brains()` / `stats()`
  * `reingest_selected(files|folders)`
  * `load_models_from_lm_studio()`

---

## 9) UX Requirements (Key Interactions)

* **On startup:** Splash with “Loading… warming caches.”
* **Ask a question:**

  * Input → **Submit** button just under input, **Ctrl+Enter** to submit.
  * Show streaming answer; show citations with clickable file path.
  * Show timing breakdown line: `Retrieval=Xs | LLM=Ys | Total=Zs`.
* **DB Search panel:** Quick filter by file type/brain; display whether a topic seems present.
* **Manage Data:** Upload files/folders; show ingest queue; delete/restore; manual “Rebuild/Update” button.
* **Settings:** Model selection (auto-discovered), depth slider (k), temperature, OCR toggle, web-fallback toggle.

---

## 10) Success Criteria & Acceptance Tests

* **Functional:**

  * Ingests >7k files (images, HTML, PDFs, Office) and builds ~30–60k chunks without errors.
  * Retrieval returns grounded answers with at least 2 citations when content exists.
  * Deleting a doc removes its chunks and it no longer appears in results.
  * Depth slider visibly changes the number/variety of cited sources.
  * Model list mirrors LM Studio `/v1/models`; selection persists for session.

* **Performance:**

  * Warm cache queries ≤10 s total on typical questions.
  * Ingestion is resumable and skips unchanged files.
  * OCR and parsing progress bars show activity with ETA.

* **Reliability:**

  * Concurrent ingestion attempts are prevented; GUI shows an explanatory message.
  * Schema mismatch produces a clear migration instruction without corrupting data.

**Acceptance test examples** (write BDD/Gherkin if helpful):

1. *Given* a folder with 1,000 images and 100 PDFs, *when* I run ingestion, *then* OCR completes and 10k+ chunks are created and persisted.
2. *Given* two similar questions at different depth levels, *then* the number of retrieved chunks and citations differs according to k.
3. *Given* a doc is deleted via GUI, *then* its chunks disappear from results, and a restore re-adds them on the next ingest.

---

## 11) Risks & Mitigations

* **DB locks in SQLite:** Use a single-writer file lock; serialize writes; keep batch sizes moderate.
* **OCR slowness:** Tesseract is CPU-bound; add parallelism and allow per-type toggles; cache results.
* **HTML junk text:** Use readability; block common boilerplate; allow per-site rules (config).
* **Model latency:** Encourage smaller local models for responsiveness; allow depth-k tuning.

---

## 12) Deliverables SpecKit Should Produce

* **PRD** covering the above scope, flows, and acceptance criteria.
* **System Architecture** (dataflow, ingestion pipeline, retriever/LLM call graph, local-only boundaries).
* **Module/API Contracts** for ingestion, vector backend abstraction, query/answer pipeline, admin ops.
* **Data Model** schema with field definitions and indices.
* **UI Wireframes** (Streamlit layout) for: Chat, DB Search, Manage Data, Settings, Splash.
* **Test Plan** (functional + performance), including sample corpora and timing targets.
* **Migration Plan** (future pgvector), and a brief **Security/Privacy** note for air-gapped mode.
* **Operational Playbook** (how to rebuild DB, backup/restore, rotate logs, troubleshoot).

---

## 13) Glossary

* **Brain:** A namespace/tag representing a domain (e.g., `survalent`, `networking`, `finance`).
* **Chunk:** A semantically coherent text block used for retrieval.
* **Fingerprint:** File-level SHA-256 to detect changes and avoid re-ingestion.
* **RAG:** Retrieval-Augmented Generation — retrieve chunks then generate an answer with citations.

---

**End of prompt.**
