import os
import json

# Base paths & constants
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
SCHEMA_VERSION = "0.7.0"

# Define the standardized metadata schema keys
METADATA_KEYS = [
    "file_name",
    "file_path",
    "file_type",
    "domain",
    "page_number",
    "chunk_index",
    "timestamp",
    "chunk_text",
]

def ensure_structure():
    """
    Ensure the folder structure is present.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    # Optionally, create a schema version file to detect migrations
    version_file = os.path.join(DB_DIR, "schema_version.json")
    if not os.path.exists(version_file):
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump({"version": SCHEMA_VERSION}, f)

def detect_domain(file_path: str) -> str:
    """
    Heuristic to classify domain (e.g. “survalent”, “networking”, “watchguard”) based on path or filename.
    """
    lower = file_path.lower()
    if "survalent" in lower:
        return "survalent"
    if "watchguard" in lower or "wg" in lower:
        return "watchguard"
    if "network" in lower or "cisco" in lower or "router" in lower:
        return "networking"
    return "general"
