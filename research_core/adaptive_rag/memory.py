import json
from pathlib import Path
from datetime import datetime

from .state import AdaptiveRAGState

print("memory.py loaded")

MEMORY_FILE = Path(__file__).resolve().parent / "adaptive_rag_memory.json"


def save_memory_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    notes = state.get("memory_notes", [])

    memory_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_query": state.get("user_query", ""),
        "validated_query": state.get("validated_query", ""),
        "refined_query": state.get("refined_query", ""),
        "retrieval_queries": state.get("retrieval_queries", []),
        "retrieved_count": len(state.get("retrieved_documents", [])),
        "relevant_count": len(state.get("relevant_documents", [])),
        "confidence_note": state.get("confidence_note", ""),
        "final_answer_preview": state.get("final_answer", "")[:300],
    }

    try:
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        data.append(memory_entry)

        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data[-20:], f, indent=2, ensure_ascii=False)

        notes.append("Workflow memory saved successfully.")
    except Exception as e:
        notes.append(f"Memory save error: {str(e)}")

    state["memory_notes"] = notes
    return state