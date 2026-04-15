# from typing import TypedDict, List, Dict, Any


# class AdaptiveRAGState(TypedDict, total=False):
#     user_query: str
#     validated_query: str
#     refined_query: str
#     retrieval_queries: List[str]
#     query_variants: List[str]
#     retrieved_documents: List[Dict[str, Any]]
#     relevant_documents: List[Dict[str, Any]]
#     discarded_documents: List[Dict[str, Any]]
#     final_answer: str
#     confidence_note: str
#     retry_count: int
#     needs_refinement: bool
#     memory_notes: List[str]
#     retrieval_error: str
#     refinement_reason: str


# ---------------------------------------------------------------------------------------------
# --------------------------------------------------------------------

from typing import TypedDict, List, Dict, Any


class AdaptiveRAGState(TypedDict, total=False):
    # Input
    user_query: str

    # Step 1 - Query Validation
    validated_query: str
    needs_refinement: bool

    # Step 2 - Query Reformulation
    refined_query: str
    refinement_reason: str

    # Step 3 - Research Paper Retrieval
    retrieval_queries: List[str]
    retrieved_documents: List[Dict[str, Any]]
    retrieval_error: str

    # Step 4 - Relevance Filtering
    relevant_documents: List[Dict[str, Any]]
    discarded_documents: List[Dict[str, Any]]

    # Step 5 - Evidence Extraction
    evidence_blocks: List[Dict[str, Any]]

    # Step 6 - Citation-aware Output
    final_answer: str
    confidence_note: str

    # Meta
    retry_count: int
    memory_notes: List[str]
    query_variants: List[str]