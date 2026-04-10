from typing import TypedDict, List, Dict, Any


class AdaptiveRAGState(TypedDict, total=False):
    user_query: str
    validated_query: str
    refined_query: str
    retrieval_queries: List[str]
    query_variants: List[str]
    retrieved_documents: List[Dict[str, Any]]
    relevant_documents: List[Dict[str, Any]]
    discarded_documents: List[Dict[str, Any]]
    final_answer: str
    confidence_note: str
    retry_count: int
    needs_refinement: bool
    memory_notes: List[str]
    retrieval_error: str
    refinement_reason: str