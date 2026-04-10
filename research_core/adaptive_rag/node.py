from .state import AdaptiveRAGState

print("node.py loaded")


def _clean_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _is_too_short(query: str) -> bool:
    return len(query.split()) < 3


def _make_retrieval_friendly_query(query: str) -> str:
    q = query.strip()

    replacements = {
        "what is": "",
        "what are": "",
        "tell me about": "",
        "can you explain": "",
        "explain": "",
        "how does": "",
        "how do": "",
        "how is": "",
        "give me information about": "",
    }

    lowered = q.lower()
    for phrase in replacements:
        if lowered.startswith(phrase):
            q = q[len(phrase):].strip()
            break

    q = q.strip(" ?.,")
    q = _clean_text(q)
    return q


def validate_query_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    user_query = state.get("user_query", "")
    cleaned_query = _clean_text(user_query)

    if not cleaned_query:
        state["validated_query"] = ""
        state["needs_refinement"] = False
        state["confidence_note"] = "No valid query was provided."
        return state

    validated_query = _make_retrieval_friendly_query(cleaned_query)
    if not validated_query:
        validated_query = cleaned_query

    state["validated_query"] = validated_query

    notes = state.get("memory_notes", [])
    notes.append(f"Validated query prepared: {validated_query}")

    if _is_too_short(validated_query):
        notes.append("Query is short; retrieval may need refinement.")
        state["needs_refinement"] = True
    else:
        state["needs_refinement"] = False

    state["memory_notes"] = notes
    return state


def refine_query_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    base_query = state.get("validated_query") or state.get("user_query", "")
    base_query = _clean_text(base_query)

    if not base_query:
        state["refined_query"] = ""
        return state

    base_lower = base_query.lower()
    expansion_terms = []

    if "rag" in base_lower or "retrieval augmented generation" in base_lower:
        expansion_terms.extend(["retrieval augmented generation", "knowledge-intensive tasks"])

    if "agent" in base_lower or "agentic" in base_lower:
        expansion_terms.extend(["agentic workflow", "reasoning pipeline"])

    if "research" not in base_lower:
        expansion_terms.append("research")

    if "papers" not in base_lower and "literature" not in base_lower:
        expansion_terms.append("academic papers")

    expanded_parts = [base_query] + expansion_terms
    refined_query = ", ".join(dict.fromkeys(expanded_parts))
    refined_query = _clean_text(refined_query)

    state["refined_query"] = refined_query

    notes = state.get("memory_notes", [])
    notes.append(f"Query refined for retrieval: {refined_query}")
    state["memory_notes"] = notes
    state["needs_refinement"] = False

    return state


def generate_answer_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    relevant_docs = state.get("relevant_documents", [])
    validated_query = state.get("validated_query") or state.get("user_query", "")

    if not relevant_docs:
        state["final_answer"] = (
            f"Research question: {validated_query}\n\n"
            "I could not find enough relevant research documents to produce a grounded answer.\n"
            "Try refining the query or using more specific research terms."
        )
        state["confidence_note"] = "Low confidence due to limited relevant evidence."
        return state

    top_docs = relevant_docs[:3]

    evidence_lines = []
    source_lines = []

    for idx, doc in enumerate(top_docs, start=1):
        title = doc.get("title", "Untitled")
        abstract = _clean_text(doc.get("abstract", ""))
        abstract_short = abstract[:300] + ("..." if len(abstract) > 300 else "")
        label = doc.get("relevance_label", "unknown")
        score = doc.get("relevance_score", 0)
        link = doc.get("link", "")

        evidence_lines.append(
            f"{idx}. {title}\n"
            f"   Relevance: {label} (score: {score})\n"
            f"   Evidence: {abstract_short}"
        )

        if link:
            source_lines.append(f"{idx}. {title} - {link}")
        else:
            source_lines.append(f"{idx}. {title}")

    answer = (
        f"Research question: {validated_query}\n\n"
        "Grounded answer:\n"
        "Based on the most relevant retrieved papers, the topic appears in current academic literature. "
        "The following evidence was selected from retrieved sources whose titles and abstracts align with the query.\n\n"
        "Key evidence from retrieved papers:\n"
        + "\n\n".join(evidence_lines)
        + "\n\nSources:\n"
        + "\n".join(source_lines)
    )

    state["final_answer"] = answer

    high_relevance_count = sum(
        1 for d in relevant_docs if d.get("relevance_label") == "highly relevant"
    )

    if high_relevance_count >= 2:
        state["confidence_note"] = "Moderate confidence based on multiple highly relevant papers."
    else:
        state["confidence_note"] = "Preliminary confidence based on limited but relevant retrieved evidence."

    return state
