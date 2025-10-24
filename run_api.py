# run_api.py
import uvicorn
from pathlib import Path

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
        reload_excludes=["cogito-ui/**", "**/node_modules/**"],
    )
