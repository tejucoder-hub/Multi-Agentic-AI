from .state import AdaptiveRAGState

print("grading.py loaded")


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been",
    "of", "in", "on", "for", "to", "and", "or", "with", "by", "about",
    "how", "what", "why", "when", "where", "which", "used", "use",
    "into", "from", "that", "this", "these", "those", "as", "at",
    "it", "its", "than", "then", "such", "their", "them", "can",
    "could", "would", "should", "do", "does", "did"
}


TERM_EQUIVALENTS = {
    "rag": ["retrieval augmented generation"],
    "llm": ["large language model"],
    "nlp": ["natural language processing"],
    "ai": ["artificial intelligence"],
    "ml": ["machine learning"],
    "agentic": ["agent", "autonomous", "ai agent"],
}


def _normalise_text(text: str) -> str:
    cleaned = (text or "").strip().lower()
    for ch in [",", ".", ":", ";", "(", ")", "[", "]", "{", "}", "/", "\\", "-", "_", "?", "!", "\"", "'"]:
        cleaned = cleaned.replace(ch, " ")
    return " ".join(cleaned.split())


def _tokenise(text: str):
    cleaned = _normalise_text(text)
    tokens = [t for t in cleaned.split() if t and t not in STOPWORDS and len(t) > 2]
    return tokens


def _expand_query_terms(query_tokens):
    expanded = set(query_tokens)
    for token in list(query_tokens):
        if token in TERM_EQUIVALENTS:
            for phrase in TERM_EQUIVALENTS[token]:
                for part in _tokenise(phrase):
                    expanded.add(part)
    return expanded


def _score_document(query: str, document: dict) -> dict:
    query_text = _normalise_text(query)
    query_tokens = set(_tokenise(query))
    expanded_query_tokens = _expand_query_terms(query_tokens)

    title = document.get("title", "")
    abstract = document.get("abstract", "")

    title_text = _normalise_text(title)
    abstract_text = _normalise_text(abstract)

    title_tokens = set(_tokenise(title))
    abstract_tokens = set(_tokenise(abstract))
    doc_tokens = title_tokens.union(abstract_tokens)

    overlap = expanded_query_tokens.intersection(doc_tokens)
    title_overlap = expanded_query_tokens.intersection(title_tokens)
    abstract_overlap = expanded_query_tokens.intersection(abstract_tokens)

    score = 0

    score += len(title_overlap) * 3
    score += len(abstract_overlap) * 1

    if query_text and query_text in title_text:
        score += 6
    elif query_text and query_text in abstract_text:
        score += 3

    if len(title_overlap) >= 2:
        score += 2

    if expanded_query_tokens and len(overlap) >= max(2, len(expanded_query_tokens) // 3):
        score += 2

    graded_doc = dict(document)
    graded_doc["relevance_score"] = score
    graded_doc["matched_terms"] = sorted(list(overlap))

    if score >= 8:
        graded_doc["relevance_label"] = "highly relevant"
    elif score >= 4:
        graded_doc["relevance_label"] = "partially relevant"
    else:
        graded_doc["relevance_label"] = "not relevant"

    return graded_doc


def grade_documents_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    print("grade_documents_node called")

    query = (
        state.get("refined_query")
        or state.get("validated_query")
        or state.get("user_query", "")
    ).strip()

    retrieved_docs = state.get("retrieved_documents", [])

    if not retrieved_docs or not query:
        state["relevant_documents"] = []
        state["discarded_documents"] = []
        state["needs_refinement"] = True
        state["refinement_reason"] = "no_documents"
        return state

    graded_documents = [_score_document(query, doc) for doc in retrieved_docs]
    graded_documents.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    relevant_documents = []
    discarded_documents = []

    for doc in graded_documents:
        if doc.get("relevance_label") in {"highly relevant", "partially relevant"}:
            relevant_documents.append(doc)
        else:
            discarded_documents.append(doc)

    state["relevant_documents"] = relevant_documents
    state["discarded_documents"] = discarded_documents

    high_count = sum(1 for d in relevant_documents if d.get("relevance_label") == "highly relevant")
    partial_count = sum(1 for d in relevant_documents if d.get("relevance_label") == "partially relevant")

    if high_count == 0 and partial_count < 2:
        state["needs_refinement"] = True
        state["refinement_reason"] = "low_relevance"
    else:
        state["needs_refinement"] = False
        state["refinement_reason"] = ""

    notes = state.get("memory_notes", [])
    notes.append(
        f"Graded {len(graded_documents)} document(s): "
        f"{high_count} highly relevant, {partial_count} partially relevant, "
        f"{len(discarded_documents)} not relevant."
    )
    state["memory_notes"] = notes

    return state