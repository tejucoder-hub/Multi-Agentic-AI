import json
from pathlib import Path
from typing import List


def load_json_documents(file_path: str | Path) -> List[str]:
    path = Path(file_path)
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]

        return []
    except Exception:
        return []


def load_text_documents_from_folder(folder_path: str | Path) -> List[str]:
    folder = Path(folder_path)
    if not folder.exists():
        return []

    docs: List[str] = []
    for file in folder.glob("*.txt"):
        try:
            text = file.read_text(encoding="utf-8").strip()
            if text:
                docs.append(text)
        except Exception:
            continue

    return docs