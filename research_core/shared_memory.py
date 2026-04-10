import json
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
MEMORY_FILE = CURRENT_DIR / "shared_memory_store.json"


class SharedMemory:
    def __init__(self, file_path: Path | None = None):
        self.file_path = file_path or MEMORY_FILE
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.file_path.exists():
            self.write(
                {
                    "queries": [],
                    "refined_queries": [],
                    "retrieved_docs": [],
                    "answers": [],
                    "relevance_scores": [],
                }
            )

    def read(self) -> Dict[str, List[Any]]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "queries": [],
                "refined_queries": [],
                "retrieved_docs": [],
                "answers": [],
                "relevance_scores": [],
            }

    def write(self, data: Dict[str, List[Any]]) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def append(self, key: str, value: Any) -> None:
        data = self.read()
        if key not in data:
            data[key] = []
        data[key].append(value)
        self.write(data)

    def clear(self) -> None:
        self.write(
            {
                "queries": [],
                "refined_queries": [],
                "retrieved_docs": [],
                "answers": [],
                "relevance_scores": [],
            }
        )