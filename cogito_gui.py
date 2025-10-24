# -*- coding: utf-8 -*-
# cogito_gui.py — City of Ocala's Electric AI Assistant
# - Clean sidebar UI (standardized buttons, expanders)
# - Loader button (incremental ingest) with live log + auto DB refresh
# - LM Studio Chat (langchain_openai), Chroma (langchain_chroma), MiniLM embeddings
# - Fast MMR retriever (lazy)
# - Splash status (no blank page)
# - Safe console logging (no UnicodeEncodeError on Windows)
# - Typo autocorrect + Top Sources expander
# - Ctrl+Enter to submit; ↑/↓ cycles last 20 prompts
# - FIX: no direct mutation of the text input after instantiation
# - All emoji rendered via \N{...} Unicode names to prevent wrap/encoding issues

import os
import sys
import json
import time
import locale
import re
import difflib
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# ----------------------------
# Project paths & settings
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parent
DB_DIR = str(ROOT_DIR / "chroma_db")
SETTINGS_PATH = ROOT_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "lm_studio_url": "http://localhost:1234",
    "default_model": "",
    "temperature": 0.3,
    "top_k_max": 16,
    "max_tokens": 768,
    "streaming": False,
}

def load_settings() -> Dict[str, Any]:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        s = DEFAULT_SETTINGS.copy()
        s.update(data or {})
        return s
    except Exception:
        return DEFAULT_SETTINGS.copy()

settings = load_settings()

LM_STUDIO_URL = settings.get("lm_studio_url", "http://localhost:1234").rstrip("/")
LM_STUDIO_API_KEY = "lm-studio"  # placeholder is fine for LM Studio
TOP_K_UI_MAX = int(settings.get("top_k_max", 16))
DEFAULT_MODEL = settings.get("default_model") or ""
DEFAULT_TEMPERATURE = float(settings.get("temperature", 0.3))
DEFAULT_MAX_TOKENS = int(settings.get("max_tokens", 768))
DEFAULT_STREAMING = bool(settings.get("streaming", False))

# ----------------------------
# Safe console logging (no emoji to console)
# ----------------------------
_EMOJI_RE = re.compile(
    "["  # strip symbols that break cp1252 consoles
    "\U0001F300-\U0001FAD6"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U00002300-\U000023FF"
    "]+",
    flags=re.UNICODE,
)

def _sanitize_console(s: str) -> str:
    return _EMOJI_RE.sub("", s)

def _safe_print(msg: str) -> None:
    enc = (sys.stdout.encoding or locale.getpreferredencoding(False) or "utf-8")
    out = (_sanitize_console(msg) + "\n").encode(enc, errors="replace")
    try:
        sys.stdout.buffer.write(out)
    except Exception:
        try:
            sys.stdout.write(_sanitize_console(msg) + "\n")
        except Exception:
            pass

def log_event(msg: str) -> None:
    _safe_print(msg)
    st.session_state.setdefault("log", []).append(f"{time.strftime('%H:%M:%S')} {msg}")

# ----------------------------
# Utilities
# ----------------------------
def autodiscover_models(timeout_s: int = 3) -> List[str]:
    try:
        url = f"{LM_STUDIO_URL}/v1/models"
        resp = requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
        return sorted(ids)
    except Exception as e:
        log_event(f"Model discovery failed: {e}")
        return []

# ----------------------------
# Lazy resources (cached)
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_db():
    embeddings = get_embeddings()
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return db

@st.cache_resource(show_spinner=False)
def get_retriever(k: int = 8):
    db = get_db()
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": max(2, min(int(k), TOP_K_UI_MAX)),
            "fetch_k": max(12, int(k) * 3),
            "lambda_mult": 0.5,
        },
    )
    return retriever

# ----------------------------
# Query autocorrect (DB-driven + vendor dictionary)
# ----------------------------
VENDOR_HINTS = {
    "survalent": ["Survalent", "SurvalentONE", "Survalent One"],
    "schweitzer": ["Schweitzer", "SEL", "SEL-"],
    "watchguard": ["WatchGuard"],
    "sel": ["SEL", "Schweitzer"],
    "cisco": ["Cisco"],
    "selinc": ["SEL", "Schweitzer"],
}

def _tokenize_for_match(s: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9\-\_]+", s)

def _build_vocab_from_db(db, max_items: int = 1500) -> List[str]:
    vocab: set[str] = set()
    try:
        coll = db._collection  # type: ignore (private, but stable enough)
        chunk = 300
        fetched = 0
        offset = 0
        while fetched < max_items:
            batch = coll.get(include=["metadatas"], limit=min(chunk, max_items - fetched), offset=offset)
            metas = (batch or {}).get("metadatas") or []
            if not metas:
                break
            for md in metas:
                if not md:
                    continue
                src = md.get("source") or md.get("source_path") or ""
                title = md.get("title") or ""
                for s in (src, title):
                    for tok in _tokenize_for_match(str(s)):
                        if 3 <= len(tok) <= 64:
                            vocab.add(tok)
            fetched += len(metas)
            offset += len(metas)
            if len(metas) == 0:
                break
    except Exception:
        pass
    for vs in VENDOR_HINTS.values():
        for v in vs:
            vocab.add(v)
    return sorted(vocab, key=str.lower)

def _autocorrect_query(raw_query: str, vocab: List[str], cutoff: float = 0.86) -> Tuple[str, Dict[str, str]]:
    words = re.findall(r"\b\w+\b", raw_query)
    replacements: Dict[str, str] = {}
    corrected_words = []
    for w in words:
        w_lc = w.lower()
        best = None
        if len(w) >= 5:
            for key, vals in VENDOR_HINTS.items():
                if w_lc.startswith(key) or w_lc in vals or any(w_lc == v.lower() for v in vals):
                    best = vals[0]
                    break
            if not best and vocab:
                matches = difflib.get_close_matches(w, vocab, n=1, cutoff=cutoff)
                if matches:
                    best = matches[0]
        corrected_words.append(best if best else w)
        if best and best != w:
            replacements[w] = best
    corrected = raw_query
    for src, dst in replacements.items():
        corrected = re.sub(rf"\b{re.escape(src)}\b", dst, corrected)
    return corrected, replacements

# ----------------------------
# LM Studio client & prompt
# ----------------------------
def make_llm(model_id: str, temperature: float, max_tokens: int, streaming: bool):
    return ChatOpenAI(
        api_key=LM_STUDIO_API_KEY,
        base_url=f"{LM_STUDIO_URL}/v1",
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=90,
        max_retries=0,
        streaming=streaming,
    )

SYSTEM_PROMPT = (
    "You are Cogito, a local, privacy-first RAG assistant for Ocala Electric (OT focus). "
    "Use only the retrieved context to answer. Be concise and practical. "
    "When possible, include a short bullet list of citations with file names and page numbers. "
    "If the context is insufficient, say so explicitly."
)

def build_messages(question: str, docs: List[Any]) -> List[Any]:
    snippets = []
    for i, d in enumerate(docs, 1):
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source") or meta.get("source_path") or "unknown"
        page = meta.get("page") if "page" in meta else meta.get("page_index")
        head = f"[{i}] {src}" + (f" (page {page})" if page is not None else "")
        content = (getattr(d, "page_content", "") or "").strip()[:1200]
        snippets.append(f"{head}\n{content}")
    ctx = "\n\n".join(snippets)
    user = (
        f"Question:\n{question}\n\n"
        f"Use only the provided context. If uncertain, ask for more documents.\n\n"
        f"Context:\n{ctx}"
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]

# ----------------------------
# Page config & CSS for standardized buttons
# ----------------------------
st.set_page_config(page_title="City of Ocala's Electric AI Assistant", layout="wide")
st.markdown("""
<style>
/* Sidebar button styling */
section[data-testid="stSidebar"] div.stButton > button {
  width: 100% !important;
  border-radius: 8px !important;
  padding: 0.6rem 0.9rem !important;
  border: 1px solid rgba(0,0,0,0.1) !important;
  background: #155E75 !important;       /* teal-800 */
  color: white !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
  background: #0E7490 !important;       /* teal-700 */
}
/* Compact expanders in the sidebar */
section[data-testid="stSidebar"] div[role="button"][data-baseweb="accordion"] {
  border-radius: 8px !important;
}
section[data-testid="stSidebar"] .block-container > div {
  margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

st.title("\N{HIGH VOLTAGE SIGN} City of Ocala's Electric AI Assistant")

# ---------- Prompt history (last 20) ----------
if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = []
if "prompt_history_idx" not in st.session_state:
    st.session_state["prompt_history_idx"] = -1  # -1 means "new"

def _remember_prompt(p: str):
    p = (p or "").strip()
    if not p:
        return
    hist = st.session_state["prompt_history"]
    if len(hist) == 0 or hist[-1] != p:
        hist.append(p)
        if len(hist) > 20:
            del hist[:-20]
    st.session_state["prompt_history_idx"] = -1

# ----------------------------
# Immediate splash
# ----------------------------
with st.status("\N{BRAIN} AI Loading... preparing UI", expanded=True) as status:
    status.update(label="Initializing UI elements...")

    # Layout skeleton
    col_left, col_right = st.columns([3, 2], gap="large")
    with col_left:
        st.subheader("Chat")

        # ---- Prompt input state handling (no direct mutation after widget) ----
        query_key = "prompt_input"
        # If we scheduled a set/clear from the previous run, apply it BEFORE creating the widget
        if st.session_state.get("_set_prompt_to") is not None:
            st.session_state[query_key] = st.session_state["_set_prompt_to"]
            st.session_state["_set_prompt_to"] = None
        # Ensure the key exists
        if query_key not in st.session_state:
            st.session_state[query_key] = ""

        # Text area widget (Ctrl+Enter & history wired via JS below)
        query = st.text_area(
            "Your question",
            key=query_key,
            value=st.session_state[query_key],
            height=120,
            help="Ctrl+Enter to send. Up/Down to cycle recent prompts (last 20)."
        )

        submit = st.button("Submit", type="primary", use_container_width=True, key="submit_btn")

        progress_bar = st.progress(0.0)
        stage_status = st.empty()
        answer_box = st.empty()
        sources_expander = st.expander("\N{PAGE FACING UP} Top Sources (k)", expanded=False)
        sources_box = sources_expander.empty()

    with col_right:
        st.subheader("Debug console")
        if 'log' not in st.session_state:
            st.session_state['log'] = []
        log_box = st.empty()
        timing_box = st.empty()

    # Sidebar layout
    status.update(label="Configuring sidebar...")
    with st.sidebar:
        # === Always-visible controls ===
        st.subheader("Model & Generation")
        discovered = autodiscover_models()
        if discovered:
            default_idx = 0
            if DEFAULT_MODEL and DEFAULT_MODEL in discovered:
                default_idx = discovered.index(DEFAULT_MODEL)
            model_choice = st.selectbox("Model", options=discovered, index=default_idx)
        else:
            st.warning("Could not discover models. Ensure LM Studio is running.")
            model_choice = st.text_input("Model ID (manual)", value=(DEFAULT_MODEL or ""))

        temperature = st.slider("Temperature", 0.0, 1.5, float(DEFAULT_TEMPERATURE), 0.05)
        max_tokens = st.slider("Max tokens", 128, 4096, int(DEFAULT_MAX_TOKENS), 64)
        streaming_enabled = st.toggle(
            "Stream tokens (experimental)",
            value=bool(DEFAULT_STREAMING),
            help="If you see empty answers, turn this off."
        )

        st.subheader("Retrieval")
        depth = st.slider("Answer Depth (k)", 1, int(TOP_K_UI_MAX), min(8, int(TOP_K_UI_MAX)))

        st.subheader("Data")
        run_loader_now = st.button("\N{INBOX TRAY} Check for New / Changed Files", use_container_width=True)
        loader_log_box = st.empty()

        # === Collapsible advanced sections ===
        with st.expander("\N{CARD INDEX DIVIDERS} Database Options", expanded=False):
            st.caption(f"DB dir: `{DB_DIR}`")
            refresh_db = st.button("Refresh DB Info", use_container_width=True)
            db_info_box = st.empty()

        with st.expander("\N{DESKTOP COMPUTER} Server Options", expanded=False):
            colA, colB = st.columns(2, gap="small")
            with colA:
                if st.button("\N{ANTICLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS} Restart App", use_container_width=True):
                    st.cache_resource.clear()
                    st.rerun()
            with colB:
                if st.button("\N{BROOM} Clear Caches", use_container_width=True):
                    st.cache_resource.clear()
                    st.success("Caches cleared. They will rebuild on next action.")

    status.update(label="UI ready. You can interact now.")
    status.update(state="complete")

# --- JS: Ctrl+Enter submit + prompt history with Up/Down (visual only) ---
hist_json = json.dumps(st.session_state["prompt_history"])
st.markdown(f"""
<script>
(function() {{
  const findTextArea = () => document.querySelector('textarea');
  const findSubmit = () => Array.from(document.querySelectorAll('button')).find(b => /submit/i.test(b.innerText));
  let hist = {hist_json};
  let idx = -1;

  function applyValue(val) {{
    const ta = findTextArea();
    if (!ta) return;
    ta.value = val || "";
    ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
  }}

  document.addEventListener('keydown', function(e) {{
    const ta = findTextArea();
    if (!ta || document.activeElement !== ta) return;

    // Ctrl+Enter to submit
    if (e.ctrlKey && e.key === 'Enter') {{
      const btn = findSubmit();
      if (btn) btn.click();
      e.preventDefault();
      return;
    }}

    // Up/Down to navigate history
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {{
      if (hist.length === 0) return;
      if (idx === -1) idx = hist.length;
      if (e.key === 'ArrowUp') idx = Math.max(0, idx - 1);
      if (e.key === 'ArrowDown') idx = Math.min(hist.length, idx + 1);
      if (idx >= 0 && idx < hist.length) {{
        applyValue(hist[idx]);
      }} else {{
        applyValue("");
      }}
      e.preventDefault();
    }}
  }});
}})();
</script>
""", unsafe_allow_html=True)

# Keep debug console alive
log_box.code("\n".join(st.session_state['log'][-200:]) or "(no log yet)", language="text")

# ----------------------------
# Loader integration (incremental ingest) — force UTF-8 child I/O
# ----------------------------
def run_loader_with_live_log():
    """
    Runs `python cogito_loader.py --progress` and streams output into the GUI.
    Forces the child process to use UTF-8 so emoji/Unicode prints don't crash on Windows.
    On success, clears caches so the GUI sees the updated DB immediately.
    """
    # Force UTF-8 in the child Python (covers stdout/stderr and text mode)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [
        sys.executable,
        "-X", "utf8",   # belt-and-suspenders
        "-u",
        str(ROOT_DIR / "cogito_loader.py"),
        "--progress",
    ]

    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW  # hide console on Windows

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            env=env,                       # ensure UTF-8
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",              # decode pipe as UTF-8
            errors="replace",
            creationflags=creationflags,
            bufsize=1,
        )
    except Exception as e:
        loader_log_box.error(f"Failed to start loader: {e}")
        return

    log_lines: List[str] = []
    progress_val = 0.0
    with st.status("Ingestion running (incremental)", expanded=True) as loader_status:
        pb = st.progress(progress_val)
        last_ui = time.time()

        while True:
            if proc.poll() is not None:
                tail = proc.stdout.read() if proc.stdout else ""
                if tail:
                    log_lines.extend(tail.splitlines())
                break

            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                log_lines.append(line.rstrip("\n"))
                now = time.time()
                if now - last_ui > 0.15:
                    progress_val = min(0.95, progress_val + 0.01)
                    pb.progress(progress_val)
                    loader_log_box.code("\n".join(log_lines[-500:]) or "(no output yet)")
                    last_ui = now
            else:
                time.sleep(0.05)

        rc = proc.returncode or 0
        if rc == 0:
            pb.progress(1.0)
            loader_status.update(label="Ingestion complete (incremental). Refreshing DB...")
            loader_log_box.code("\n".join(log_lines[-500:]) or "(no output)")
            # Clear in-process caches so new docs are visible immediately
            st.cache_resource.clear()
            try:
                db = get_db()
                with st.sidebar:
                    with st.expander("\N{CARD INDEX DIVIDERS} Database Options", expanded=False):
                        st.caption(f"DB dir: `{DB_DIR}`")
                        try:
                            cols = [c.name for c in db._client.list_collections()]
                            st.write({"collections": cols})
                        except Exception:
                            st.write("(collections unavailable)")
            except Exception:
                pass
            loader_status.update(state="complete")
            st.success("Vector DB reloaded in the app.")
        else:
            loader_status.update(label=f"Loader exited with code {rc}", state="error")
            loader_log_box.code("\n".join(log_lines[-500:]))

if run_loader_now:
    run_loader_with_live_log()

# ----------------------------
# DB Info (on demand)
# ----------------------------
if 'refresh_db' not in locals():
    refresh_db = False
if refresh_db:
    t0 = time.time()
    try:
        progress_bar.progress(0.1)
        stage_status.markdown("Loading database info...")
        db = get_db()
        collections = [c.name for c in db._client.list_collections()]
        with st.sidebar:
            with st.expander("\N{CARD INDEX DIVIDERS} Database Options", expanded=True):
                st.caption(f"DB dir: `{DB_DIR}`")
                st.write({"collections": collections})
        stage_status.markdown(f"Database info loaded in {round(time.time() - t0, 2)}s")
        progress_bar.progress(0.3)
    except Exception as e:
        with st.sidebar:
            with st.expander("\N{CARD INDEX DIVIDERS} Database Options", expanded=True):
                st.error(f"DB info unavailable: {e}")
        stage_status.markdown("DB info failed to load.")
        progress_bar.progress(0.0)

# ----------------------------
# RAG execution
# ----------------------------
def get_vocab_cached():
    if "db_vocab" not in st.session_state:
        try:
            st.session_state["db_vocab"] = _build_vocab_from_db(get_db(), max_items=1500)
        except Exception:
            st.session_state["db_vocab"] = []
    return st.session_state["db_vocab"]

# Detect submission via button OR Ctrl+Enter (the JS clicks the same button)
submitted_now = bool(submit)
user_text = (st.session_state.get("prompt_input") or "").strip()

if submitted_now and user_text:
    # Remember the prompt. DO NOT mutate the widget value directly here.
    _remember_prompt(user_text)
    # Schedule clearing the input on the next rerun (safe timing)
    st.session_state["_set_prompt_to"] = ""

    query = user_text  # keep local copy for this run
    total_start = time.time()
    try:
        progress_bar.progress(0.05)
        stage_status.markdown("Building retriever...")
        t0 = time.time()
        retriever = get_retriever(k=int(depth))
        retriever_init_s = round(time.time() - t0, 2)
        progress_bar.progress(0.2)

        vocab = get_vocab_cached()

        # 1st pass
        stage_status.markdown("Fetching relevant documents...")
        t1 = time.time()
        docs = retriever.get_relevant_documents(query)
        doc_fetch_s = round(time.time() - t1, 2)
        progress_bar.progress(0.35)

        corrected_query, replacements = query, {}

        # Autocorrect if thin hits (likely typos)
        if len(docs) == 0 or (len(docs) <= 2 and any(len(w) >= 5 for w in _tokenize_for_match(query))):
            corrected_query, replacements = _autocorrect_query(query, vocab)
            if replacements:
                stage_status.markdown(f"Did you mean: **{corrected_query}** ? Retrying retrieval...")
                t1b = time.time()
                docs2 = retriever.get_relevant_documents(corrected_query)
                if len(docs2) >= len(docs):
                    docs = docs2
                doc_fetch_s += round(time.time() - t1b, 2)

        # Show sources
        if docs:
            tops = []
            for d in docs[: max(10, int(depth))]:
                m = getattr(d, "metadata", {}) or {}
                src = m.get("source") or m.get("source_path") or "unknown"
                page = m.get("page") if "page" in m else m.get("page_index")
                tops.append(f"- {src}" + (f", page {page}" if page is not None else ""))
            sources_box.markdown("\n".join(tops) or "(no sources)")
        else:
            sources_box.markdown("(no sources)")

        progress_bar.progress(0.5)

        stage_status.markdown("Connecting to LM Studio...")
        llm = make_llm(
            model_id=(model_choice or "gpt-3.5-turbo"),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            streaming=bool(streaming_enabled),
        )

        stage_status.markdown("Generating answer...")
        t2 = time.time()
        if streaming_enabled:
            messages = build_messages(corrected_query, docs)
            streamed_text = []
            for chunk in llm.stream(messages):
                token = getattr(chunk, "content", "") or ""
                if token:
                    streamed_text.append(token)
                    answer_box.markdown("".join(streamed_text))
            answer = ("".join(streamed_text) or "").strip()
        else:
            messages = build_messages(corrected_query, docs)
            resp = llm.invoke(messages)
            answer = (getattr(resp, "content", "") or "").strip()
        llm_time_s = round(time.time() - t2, 2)
        progress_bar.progress(0.85)

        if not answer:
            candidates = []
            for d in docs[: min(5, len(docs))]:
                m = getattr(d, "metadata", {}) or {}
                src = m.get("source") or m.get("source_path") or "unknown"
                page = m.get("page") if "page" in m else m.get("page_index")
                candidates.append(f"- {src}" + (f", page {page}" if page is not None else ""))
            candidate_str = "\n".join(candidates) or "- (no candidates returned)"
            if replacements:
                repl_str = ", ".join([f"{a}->{b}" for a, b in replacements.items()])
                answer = (
                    "I could not produce an answer from the retrieved context even after autocorrect ("
                    + repl_str + ").\n\nTop candidate sources I found:\n" + candidate_str
                )
            else:
                answer = (
                    "I could not produce an answer from the retrieved context.\n\n"
                    "Top candidate sources I found:\n" + candidate_str
                )

        answer_box.markdown(answer)

        total_s = round(time.time() - total_start, 2)
        timing_box.write({
            "retriever_init_s": retriever_init_s,
            "doc_fetch_s": doc_fetch_s,
            "llm_time_s": llm_time_s,
            "total_cycle_s": total_s
        })
        log_event(f"[timings] retrieval={retriever_init_s + doc_fetch_s}s, llm={llm_time_s}s, total={total_s}s")
        progress_bar.progress(1.0)
        stage_status.markdown("Complete")

    except Exception as e:
        progress_bar.progress(1.0)
        stage_status.markdown("Error")
        answer_box.error(f"Error: {e}")
        st.exception(e)
