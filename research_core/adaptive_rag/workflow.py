from .state import AdaptiveRAGState
from .node import (
    validate_query_node,
    refine_query_node,
    generate_answer_node,
)
from .retrieval import retrieve_documents_node
from .grading import grade_documents_node
from .memory import save_memory_node

print("workflow.py loaded")


def run_adaptive_rag_workflow(user_query: str) -> AdaptiveRAGState:
    print("run_adaptive_rag_workflow started")

    state: AdaptiveRAGState = {
        "user_query": user_query,
        "validated_query": "",
        "refined_query": "",
        "retrieval_queries": [],
        "retrieved_documents": [],
        "relevant_documents": [],
        "discarded_documents": [],
        "final_answer": "",
        "confidence_note": "",
        "retry_count": 0,
        "needs_refinement": False,
        "memory_notes": [],
    }

    state = validate_query_node(state)
    state = retrieve_documents_node(state)
    state = grade_documents_node(state)

    if state.get("needs_refinement") and state.get("retry_count", 0) < 1:
        state["retry_count"] = state.get("retry_count", 0) + 1
        state = refine_query_node(state)
        state = retrieve_documents_node(state)
        state = grade_documents_node(state)

    state = generate_answer_node(state)
    state = save_memory_node(state)

    return state
