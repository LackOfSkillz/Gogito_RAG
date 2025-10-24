# -*- coding: utf-8 -*-
# cogito_query.py — RAG with autocorrect, query expansion, performance modes, and info-density scoring

from __future__ import annotations
import os, re, difflib, time, math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ROOT_DIR, "chroma_db")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234").rstrip("/")
LM_STUDIO_API_KEY = "lm-studio"  # placeholder

SYSTEM_PROMPT = (
    "You are Cogito, a local, privacy-first RAG assistant for Ocala Electric (OT focus).\n"
    "Ground answers strictly in the provided context snippets (no outside facts).\n"
    "Audience: utility OT/engineering leadership. Be crisp and practical.\n"
    "When appropriate, include a short list of citations with file names and page numbers.\n"
    "Never say you need more documents; always provide a best-effort summary of what's present."
)

# ---------------- caches ----------------
@lru_cache(maxsize=1)
def _embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def _db():
    return Chroma(persist_directory=DB_DIR, embedding_function=_embeddings())

def _retriever(k: int):
    k = max(2, min(int(k or 8), 32))
    return _db().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(12, k * 3), "lambda_mult": 0.45},
    )

def _llm(model_id: Optional[str], temperature: float, max_tokens: int, streaming: bool = False):
    return ChatOpenAI(
        api_key=LM_STUDIO_API_KEY,
        base_url=f"{LM_STUDIO_URL}/v1",
        model=model_id or "gpt-3.5-turbo",
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        timeout=90,
        max_retries=0,
        streaming=bool(streaming),
    )

# ------------- autocorrect + vocab -------------
VENDOR_HINTS = {
    "survalent": ["Survalent", "SurvalentONE", "Survalent One"],
    "schweitzer": ["Schweitzer", "SEL", "SEL-"],
    "watchguard": ["WatchGuard"],
    "sel": ["SEL", "Schweitzer"],
    "cisco": ["Cisco"],
    "selinc": ["SEL", "Schweitzer"],
}
_token = re.compile(r"[A-Za-z][A-Za-z0-9\-\_]+")

def _tokenize(s: str) -> List[str]:
    return _token.findall(s or "")

@lru_cache(maxsize=1)
def _db_vocab(max_items: int = 1500) -> List[str]:
    vocab: set[str] = set()
    try:
        coll = _db()._collection  # private, stable enough
        chunk = 300
        fetched = 0
        offset = 0
        while fetched < max_items:
            batch = coll.get(include=["metadatas"], limit=min(chunk, max_items - fetched), offset=offset)
            metas = (batch or {}).get("metadatas") or []
            if not metas: break
            for md in metas:
                if not md: continue
                for s in (str(md.get("source") or md.get("source_path") or ""), str(md.get("title") or "")):
                    for tok in _tokenize(s):
                        if 3 <= len(tok) <= 64: vocab.add(tok)
            got = len(metas); fetched += got; offset += got
            if got == 0: break
    except Exception:
        pass
    for vs in VENDOR_HINTS.values(): vocab.update(vs)
    return sorted(vocab, key=str.lower)

def _autocorrect_query(raw_query: str, cutoff: float = 0.86) -> Tuple[str, Dict[str, str]]:
    vocab = _db_vocab()
    words = re.findall(r"\b\w+\b", raw_query or "")
    replacements: Dict[str, str] = {}
    out: List[str] = []
    for w in words:
        w_lc = w.lower()
        best = None
        if len(w) >= 5:
            for key, vals in VENDOR_HINTS.items():
                if w_lc.startswith(key) or w_lc in vals or any(w_lc == v.lower() for v in vals):
                    best = vals[0]; break
            if not best and vocab:
                match = difflib.get_close_matches(w, vocab, n=1, cutoff=cutoff)
                if match: best = match[0]
        out.append(best if best else w)
        if best and best != w: replacements[w] = best
    corrected = raw_query
    for src, dst in replacements.items():
        corrected = re.sub(rf"\b{re.escape(src)}\b", dst, corrected)
    return corrected, replacements

def _expand_query(q: str) -> str:
    extras = ["SurvalentONE", "SCADA", "ADMS", "DERMS", "OMS", "FLISR", "Distribution", "Feeder", "Outage"]
    add = [e for e in extras if e.lower() not in q.lower()]
    return q + (" — " + " ".join(add[:6]) if add else "")

# ------------- info-density scoring -------------
_BOILERPLATE = re.compile(
    r"(confidential|proprietary|disclosure|without prior written|all rights reserved|confidentiality notice|copyright)",
    re.IGNORECASE,
)
_DOMAIN_BOOST_TERMS = [
    "scada","adms","derms","oms","flisr","dnp3","iccp","modbus","histori",
    "substation","distribution","feeder","restoration","switch","recloser","gis","pi","survalentone"
]

def _info_density_score(text: str, src: str) -> float:
    if not text: return -1.0
    t = text.lower()
    # base on length (log to dampen)
    base = math.log10(1 + len(t)) * 0.5
    # domain keywords boost
    boost = sum(t.count(k) for k in _DOMAIN_BOOST_TERMS) * 0.8
    # penalize boilerplate/legal
    pen = 2.0 if _BOILERPLATE.search(t) else 0.0
    # small bonus if filename looks relevant
    fn = os.path.basename(src).lower()
    name_boost = 0.6 if any(k in fn for k in ["scada","adms","derms","flisr","oms","training","manual","refresher"]) else 0.0
    return base + boost + name_boost - pen

def _rank_docs_by_density(docs: List[Any]) -> List[Any]:
    scored = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        src = md.get("source") or md.get("source_path") or "unknown"
        txt = (getattr(d, "page_content", "") or "")
        score = _info_density_score(txt, str(src))
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]

# ------------- prompting helpers -------------
def _build_context(docs: List[Any], take: int, max_chars: int) -> Tuple[str, List[Dict[str, Any]]]:
    snippets = []
    sources: List[Dict[str, Any]] = []
    for i, d in enumerate(docs[:take], 1):
        md = getattr(d, "metadata", {}) or {}
        src = md.get("source") or md.get("source_path") or "unknown"
        page = md.get("page") if "page" in md else md.get("page_index")
        head = f"[{i}] {src}" + (f" (page {page})" if page is not None else "")
        txt = (getattr(d, "page_content", "") or "").strip()
        snippets.append(f"{head}\n{txt[:max_chars]}")
        sources.append({"source": src, "page": page})
    return "\n\n".join(snippets), sources

def _messages_for_note(dtext: str, tag: str):
    user = (
        "Extract up to 2 self-contained bullets with concrete facts (no speculation). "
        "Prioritize: what it is, capabilities/modules, deployment patterns, benefits — only if present.\n\n"
        f"Excerpt ({tag}):\n{dtext[:1500]}"
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]

def _messages_for_merge(notes: List[str], question: str, style: str):
    joined = "\n".join(f"- {n}" for n in notes if n.strip())
    if style == "fast":
        target = "Write 4–6 tight bullets."
    elif style == "thorough":
        target = "Write 8–12 bullets or two concise paragraphs."
    else:
        target = "Write 6–8 bullets."
    user = (
        f"Merge these grounded notes into an executive summary. {target} "
        "Only include information supported by the notes. "
        "Do NOT say more documents are needed.\n\n"
        f"Question: {question}\n\nNotes:\n{joined}"
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]

# ------------- public API -------------
def ask(
    question: str,
    depth: int = 12,
    temperature: float = 0.25,
    max_tokens: int = 900,
    model: Optional[str] = None,
    mode: str = "balanced",  # "fast" | "balanced" | "thorough"
) -> Dict[str, Any]:

    # mode presets
    presets = {
        "fast":      {"top_docs": 6,  "note_docs": 4, "note_chars": 1200, "max_tokens": min(max_tokens, 600)},
        "balanced":  {"top_docs": 10, "note_docs": 8, "note_chars": 1500, "max_tokens": max_tokens},
        "thorough":  {"top_docs": 14, "note_docs": 12,"note_chars": 2000, "max_tokens": max(max_tokens, 1200)},
    }
    p = presets.get(mode, presets["balanced"])

    t0 = time.time()
    retriever = _retriever(depth)
    t1 = time.time()

    # initial retrieval
    docs: List[Any] = retriever.invoke(question)
    t2 = time.time()

    # retry: autocorrect + expansion if thin
    corrected_query, replacements = question, {}
    if len(docs) == 0 or (len(docs) <= 2 and any(len(w) >= 5 for w in _tokenize(question))):
        corrected_query, replacements = _autocorrect_query(question)
        q2 = corrected_query
        if q2 != question:
            docs2 = retriever.invoke(q2)
            if len(docs2) >= len(docs): docs = docs2
        if len(docs) <= 2:
            q3 = _expand_query(q2)
            if q3 != q2:
                docs3 = retriever.invoke(q3)
                if len(docs3) >= len(docs): docs = docs3
                corrected_query = q3

    # NEW: rank by info density to avoid boilerplate-heavy pages
    docs = _rank_docs_by_density(docs)

    # map–reduce summary (mode-tuned)
    llm = _llm(model, temperature, p["max_tokens"], streaming=False)

    # Map: extract short notes from top N docs
    notes: List[str] = []
    for idx, d in enumerate(docs[: p["note_docs"]], 1):
        dtext = (getattr(d, "page_content", "") or "")[: p["note_chars"]]
        if not dtext.strip(): continue
        note_resp = llm.invoke(_messages_for_note(dtext, f"doc{idx}"))
        n = (getattr(note_resp, "content", "") or "").strip()
        if n:
            for line in n.splitlines():
                line = line.strip().lstrip("-• \t")
                if line: notes.append(line)
        if mode == "fast" and len(notes) >= 10:
            break

    # Reduce: merge notes
    if notes:
        merged = llm.invoke(_messages_for_merge(notes, question, style=mode))
        answer = (getattr(merged, "content", "") or "").strip()
    else:
        # Fallback: summarize raw snippets directly
        ctx, _ = _build_context(docs, take=p["top_docs"], max_chars=p["note_chars"])
        merged = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                "Create a best-effort executive summary (bullets or two short paragraphs) strictly from this context. "
                "No statements about needing more documents.\n\n"
                f"{ctx}"
            ))
        ])
        answer = (getattr(merged, "content", "") or "").strip()

    # Sources for UI
    _, sources = _build_context(docs, take=min(p["top_docs"], max(8, depth)), max_chars=400)

    t3 = time.time()
    return {
        "answer": answer or "Summary based on limited context.",
        "sources": sources,
        "timings": {
            "retriever_init_s": round(t1 - t0, 2),
            "doc_fetch_s": round(t2 - t1, 2),
            "llm_time_s": round(t3 - t2, 2),
            "total_s": round(t3 - t0, 2),
        },
        "query_used": corrected_query,
        "replacements": replacements,
        "mode": mode,
    }

if __name__ == "__main__":
    out = ask("Give me an executive summary of Survalent.", depth=12, temperature=0.25, max_tokens=900, model=None, mode="fast")
    print(out["answer"][:500])
    print("Timings:", out["timings"])
