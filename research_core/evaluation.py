from typing import List


def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 3) -> float:
    if k <= 0:
        return 0.0

    retrieved_k = retrieved_docs[:k]
    if not retrieved_k:
        return 0.0

    relevant_set = set(relevant_docs)
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / len(retrieved_k)


def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 3) -> float:
    if not relevant_docs:
        return 0.0

    retrieved_k = retrieved_docs[:k]
    relevant_set = set(relevant_docs)
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / len(relevant_set)