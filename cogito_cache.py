# -*- coding: utf-8 -*-
# cogito_cache.py — Answer cache with hash-based lookup and invalidation

from __future__ import annotations
import os
import json
import hashlib
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DB = ROOT_DIR / ".cache" / "answer_cache.db"

# Ensure cache directory exists
CACHE_DB.parent.mkdir(parents=True, exist_ok=True)


class AnswerCache:
    """
    Cache answers keyed by (question_norm, mode, depth, model, space, chunk_ids_hash).
    Invalidate entries when any of their source chunks change.
    """

    def __init__(self, db_path: Path = CACHE_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create cache table if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS answer_cache (
                    cache_key TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    mode TEXT,
                    depth INTEGER,
                    temperature REAL,
                    max_tokens INTEGER,
                    model TEXT,
                    space TEXT,
                    chunk_ids_hash TEXT,
                    query_used TEXT,
                    replacements TEXT,
                    created_at REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_ids_hash 
                ON answer_cache(chunk_ids_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON answer_cache(created_at)
            """)
            conn.commit()

    def _normalize_question(self, question: str) -> str:
        """Normalize question for better cache hits."""
        q = question.lower().strip()
        # Remove extra whitespace
        q = " ".join(q.split())
        # Remove common punctuation variations
        q = q.rstrip("?!.")
        return q

    def _generate_cache_key(
        self,
        question: str,
        mode: str,
        depth: int,
        temperature: float,
        max_tokens: int,
        model: Optional[str],
        space: str,
        chunk_ids: List[str],
    ) -> str:
        """Generate cache key from normalized inputs."""
        q_norm = self._normalize_question(question)
        chunk_ids_sorted = sorted(chunk_ids)
        chunk_hash = hashlib.sha256(
            json.dumps(chunk_ids_sorted, sort_keys=True).encode()
        ).hexdigest()[:16]

        key_parts = [
            q_norm,
            mode,
            str(depth),
            f"{temperature:.2f}",
            str(max_tokens),
            model or "default",
            space or "default",
            chunk_hash,
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(
        self,
        question: str,
        mode: str,
        depth: int,
        temperature: float,
        max_tokens: int,
        model: Optional[str],
        space: str,
        chunk_ids: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Look up cached answer.
        Returns None if not found or invalid.
        """
        cache_key = self._generate_cache_key(
            question, mode, depth, temperature, max_tokens, model, space, chunk_ids
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT answer, sources, query_used, replacements, mode, created_at
                FROM answer_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            )
            row = cursor.fetchone()

            if row:
                # Increment hit count
                conn.execute(
                    "UPDATE answer_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (cache_key,),
                )
                conn.commit()

                return {
                    "answer": row["answer"],
                    "sources": json.loads(row["sources"]),
                    "query_used": row["query_used"],
                    "replacements": json.loads(row["replacements"] or "{}"),
                    "mode": row["mode"],
                    "cached": True,
                    "cache_age_s": round(time.time() - row["created_at"], 1),
                }

        return None

    def put(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        mode: str,
        depth: int,
        temperature: float,
        max_tokens: int,
        model: Optional[str],
        space: str,
        chunk_ids: List[str],
        query_used: str,
        replacements: Dict[str, str],
    ):
        """Store answer in cache."""
        cache_key = self._generate_cache_key(
            question, mode, depth, temperature, max_tokens, model, space, chunk_ids
        )

        chunk_ids_sorted = sorted(chunk_ids)
        chunk_hash = hashlib.sha256(
            json.dumps(chunk_ids_sorted, sort_keys=True).encode()
        ).hexdigest()[:16]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO answer_cache (
                    cache_key, question, answer, sources, mode, depth, temperature,
                    max_tokens, model, space, chunk_ids_hash, query_used, replacements,
                    created_at, hit_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    cache_key,
                    self._normalize_question(question),
                    answer,
                    json.dumps(sources),
                    mode,
                    depth,
                    temperature,
                    max_tokens,
                    model or "",
                    space or "",
                    chunk_hash,
                    query_used,
                    json.dumps(replacements),
                    time.time(),
                ),
            )
            conn.commit()

    def invalidate_by_chunk_ids(self, chunk_ids: List[str]):
        """
        Invalidate cache entries that reference any of the given chunk IDs.
        Called after ingestion when chunks are updated.
        """
        if not chunk_ids:
            return

        # Generate all possible hashes that could match
        chunk_ids_sorted = sorted(chunk_ids)
        chunk_hash = hashlib.sha256(
            json.dumps(chunk_ids_sorted, sort_keys=True).encode()
        ).hexdigest()[:16]

        with sqlite3.connect(self.db_path) as conn:
            # For now, we invalidate entries with matching hash
            # More sophisticated: track individual chunk IDs
            result = conn.execute(
                "DELETE FROM answer_cache WHERE chunk_ids_hash = ?",
                (chunk_hash,),
            )
            deleted = result.rowcount
            conn.commit()
            return deleted

    def clear_all(self):
        """Clear entire cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM answer_cache")
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Total entries
            cursor.execute("SELECT COUNT(*) as total FROM answer_cache")
            total = cursor.fetchone()["total"]

            # Total hits
            cursor.execute("SELECT SUM(hit_count) as hits FROM answer_cache")
            hits = cursor.fetchone()["hits"] or 0

            # Oldest and newest
            cursor.execute(
                "SELECT MIN(created_at) as oldest, MAX(created_at) as newest FROM answer_cache"
            )
            row = cursor.fetchone()
            oldest = row["oldest"]
            newest = row["newest"]

            # Size on disk
            size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "total_entries": total,
                "total_hits": hits,
                "hit_rate": round(hits / max(total, 1), 3),
                "oldest_entry_age_s": round(time.time() - oldest, 1) if oldest else 0,
                "newest_entry_age_s": round(time.time() - newest, 1) if newest else 0,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
            }

    def cleanup_old_entries(self, max_age_days: int = 30):
        """Remove entries older than max_age_days."""
        cutoff = time.time() - (max_age_days * 86400)
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "DELETE FROM answer_cache WHERE created_at < ?",
                (cutoff,),
            )
            deleted = result.rowcount
            conn.commit()
            return deleted


# Global cache instance
_cache_instance: Optional[AnswerCache] = None


def get_cache() -> AnswerCache:
    """Get or create the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = AnswerCache()
    return _cache_instance


if __name__ == "__main__":
    # Quick test
    cache = get_cache()
    print("Cache stats:", cache.get_stats())
