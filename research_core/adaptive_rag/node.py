# from .state import AdaptiveRAGState

# print("node.py loaded")


# def _clean_text(text: str) -> str:
#     return " ".join((text or "").strip().split())


# def _is_too_short(query: str) -> bool:
#     return len(query.split()) < 3


# def _make_retrieval_friendly_query(query: str) -> str:
#     q = query.strip()

#     replacements = {
#         "what is": "",
#         "what are": "",
#         "tell me about": "",
#         "can you explain": "",
#         "explain": "",
#         "how does": "",
#         "how do": "",
#         "how is": "",
#         "give me information about": "",
#     }

#     lowered = q.lower()
#     for phrase in replacements:
#         if lowered.startswith(phrase):
#             q = q[len(phrase):].strip()
#             break

#     q = q.strip(" ?.,")
#     q = _clean_text(q)
#     return q


# def validate_query_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
#     user_query = state.get("user_query", "")
#     cleaned_query = _clean_text(user_query)

#     if not cleaned_query:
#         state["validated_query"] = ""
#         state["needs_refinement"] = False
#         state["confidence_note"] = "No valid query was provided."
#         return state

#     validated_query = _make_retrieval_friendly_query(cleaned_query)
#     if not validated_query:
#         validated_query = cleaned_query

#     state["validated_query"] = validated_query

#     notes = state.get("memory_notes", [])
#     notes.append(f"Validated query prepared: {validated_query}")

#     if _is_too_short(validated_query):
#         notes.append("Query is short; retrieval may need refinement.")
#         state["needs_refinement"] = True
#     else:
#         state["needs_refinement"] = False

#     state["memory_notes"] = notes
#     return state


# def refine_query_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
#     base_query = state.get("validated_query") or state.get("user_query", "")
#     base_query = _clean_text(base_query)

#     if not base_query:
#         state["refined_query"] = ""
#         return state

#     base_lower = base_query.lower()
#     expansion_terms = []

#     if "rag" in base_lower or "retrieval augmented generation" in base_lower:
#         expansion_terms.extend(["retrieval augmented generation", "knowledge-intensive tasks"])

#     if "agent" in base_lower or "agentic" in base_lower:
#         expansion_terms.extend(["agentic workflow", "reasoning pipeline"])

#     if "research" not in base_lower:
#         expansion_terms.append("research")

#     if "papers" not in base_lower and "literature" not in base_lower:
#         expansion_terms.append("academic papers")

#     expanded_parts = [base_query] + expansion_terms
#     refined_query = ", ".join(dict.fromkeys(expanded_parts))
#     refined_query = _clean_text(refined_query)

#     state["refined_query"] = refined_query

#     notes = state.get("memory_notes", [])
#     notes.append(f"Query refined for retrieval: {refined_query}")
#     state["memory_notes"] = notes
#     state["needs_refinement"] = False

#     return state


# def generate_answer_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
#     relevant_docs = state.get("relevant_documents", [])
#     validated_query = state.get("validated_query") or state.get("user_query", "")

#     if not relevant_docs:
#         state["final_answer"] = (
#             f"Research question: {validated_query}\n\n"
#             "I could not find enough relevant research documents to produce a grounded answer.\n"
#             "Try refining the query or using more specific research terms."
#         )
#         state["confidence_note"] = "Low confidence due to limited relevant evidence."
#         return state

#     top_docs = relevant_docs[:3]

#     evidence_lines = []
#     source_lines = []

#     for idx, doc in enumerate(top_docs, start=1):
#         title = doc.get("title", "Untitled")
#         abstract = _clean_text(doc.get("abstract", ""))
#         abstract_short = abstract[:300] + ("..." if len(abstract) > 300 else "")
#         label = doc.get("relevance_label", "unknown")
#         score = doc.get("relevance_score", 0)
#         link = doc.get("link", "")

#         evidence_lines.append(
#             f"{idx}. {title}\n"
#             f"   Relevance: {label} (score: {score})\n"
#             f"   Evidence: {abstract_short}"
#         )

#         if link:
#             source_lines.append(f"{idx}. {title} - {link}")
#         else:
#             source_lines.append(f"{idx}. {title}")

#     answer = (
#         f"Research question: {validated_query}\n\n"
#         "Grounded answer:\n"
#         "Based on the most relevant retrieved papers, the topic appears in current academic literature. "
#         "The following evidence was selected from retrieved sources whose titles and abstracts align with the query.\n\n"
#         "Key evidence from retrieved papers:\n"
#         + "\n\n".join(evidence_lines)
#         + "\n\nSources:\n"
#         + "\n".join(source_lines)
#     )

#     state["final_answer"] = answer

#     high_relevance_count = sum(
#         1 for d in relevant_docs if d.get("relevance_label") == "highly relevant"
#     )

#     if high_relevance_count >= 2:
#         state["confidence_note"] = "Moderate confidence based on multiple highly relevant papers."
#     else:
#         state["confidence_note"] = "Preliminary confidence based on limited but relevant retrieved evidence."

#     return state


# ------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
"""
adaptive_rag/node.py

6-step pipeline matching the architecture diagram:
  Step 1 → validate_query_node       (Query Validation)
  Step 2 → refine_query_node         (Query Reformulation)
  Step 3 → [retrieval.py]            (Research Paper Retrieval)
  Step 4 → [grading.py]              (Relevance Filtering)
  Step 5 → extract_evidence_node     (Evidence Extraction)
  Step 6 → generate_answer_node      (Citation-aware Output)
"""

import os
from .state import AdaptiveRAGState

print("node.py loaded")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    return " ".join((text or "").strip().split())


def _is_short(query: str) -> bool:
    return len(query.split()) < 3


def _strip_intent(query: str) -> str:
    q = query.strip()
    prefixes = [
        "what is", "what are", "what's", "tell me about", "can you explain",
        "explain", "how does", "how do", "how is", "give me information about",
        "give me", "find me", "show me", "i want to know about",
    ]
    lowered = q.lower()
    for p in prefixes:
        if lowered.startswith(p):
            q = q[len(p):].strip()
            break
    return _clean(q.strip(" ?.,"))


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Query Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_query_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    user_query = state.get("user_query", "")
    cleaned    = _clean(user_query)

    if not cleaned:
        state["validated_query"]  = ""
        state["needs_refinement"] = False
        state["confidence_note"]  = "No query provided."
        return state

    validated = _strip_intent(cleaned) or cleaned
    state["validated_query"]  = validated
    state["needs_refinement"] = _is_short(validated)

    notes = state.get("memory_notes", [])
    notes.append(f"[Step 1] Validated: '{validated}'")
    state["memory_notes"] = notes
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Query Reformulation
# ─────────────────────────────────────────────────────────────────────────────

def refine_query_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    base = _clean(state.get("validated_query") or state.get("user_query", ""))
    if not base:
        state["refined_query"] = ""
        return state

    b = base.lower()
    extra = []

    if "rag" in b or "retrieval" in b:
        extra += ["retrieval augmented generation", "vector database"]
    elif "agent" in b or "agentic" in b:
        extra += ["autonomous agents", "multi-agent systems"]
    elif "llm" in b or "language model" in b:
        extra += ["transformer architecture", "language model training"]
    elif "vision" in b or "image" in b:
        extra += ["computer vision", "visual recognition"]
    else:
        extra += ["deep learning", "neural network"]

    refined = _clean(" ".join(dict.fromkeys([base] + extra)))
    state["refined_query"]    = refined
    state["needs_refinement"] = False

    notes = state.get("memory_notes", [])
    notes.append(f"[Step 2] Reformulated: '{refined}'")
    state["memory_notes"] = notes
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Evidence Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_evidence_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    relevant  = state.get("relevant_documents", [])
    retrieved = state.get("retrieved_documents", [])
    docs      = relevant if relevant else retrieved

    blocks = []
    for i, doc in enumerate(docs[:5], 1):
        title    = _clean(doc.get("title", "Untitled"))
        abstract = _clean(doc.get("abstract", ""))
        authors  = doc.get("authors") or []
        pub      = (doc.get("published") or "")[:10]
        link     = doc.get("link", "") or doc.get("pdf_link", "")
        label    = doc.get("relevance_label", "partially relevant")

        # Author citation style
        if len(authors) > 2:
            cite = f"{authors[0].split()[-1]} et al."
        elif len(authors) == 2:
            cite = f"{authors[0].split()[-1]} & {authors[1].split()[-1]}"
        elif len(authors) == 1:
            cite = authors[0].split()[-1]
        else:
            cite = "Unknown"

        year = pub[:4] if pub else "n.d."

        # First 2 sentences as key evidence
        sentences    = [s.strip() for s in abstract.split(". ") if len(s.strip()) > 30]
        key_evidence = ". ".join(sentences[:2]).strip()
        if key_evidence and not key_evidence.endswith("."):
            key_evidence += "."

        blocks.append({
            "index":        i,
            "title":        title,
            "cite":         cite,
            "year":         year,
            "link":         link,
            "key_evidence": key_evidence or abstract[:300],
            "full_abstract": abstract[:600],
            "label":        label,
            "authors_full": ", ".join(authors[:3]),
        })

    state["evidence_blocks"] = blocks

    notes = state.get("memory_notes", [])
    notes.append(f"[Step 5] Evidence extracted from {len(blocks)} paper(s).")
    state["memory_notes"] = notes
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Citation-aware Output
# ─────────────────────────────────────────────────────────────────────────────

def _llm_analysis(user_query: str, blocks: list) -> tuple[str, str]:
    """
    Ask LLM for ONLY:
      - 2-3 paragraph analysis with [N] inline citations
      - 1 sentence Research Takeaway
    Returns (analysis_text, takeaway_text)
    """
    import openai

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = openai.OpenAI(api_key=api_key)

    # Build numbered evidence list for the prompt
    evidence_lines = []
    for b in blocks:
        evidence_lines.append(
            f"[{b['index']}] \"{b['title']}\" — {b['cite']} ({b['year']})\n"
            f"    Finding: {b['key_evidence']}"
        )
    evidence_text = "\n\n".join(evidence_lines)

    system_prompt = """You are an expert academic research assistant.

TASK: Write a citation-aware research answer in EXACTLY this format:

---
**Overview**
[One paragraph directly answering the question. Use [N] inline citations naturally.]

**What the Research Shows**
[1-2 paragraphs synthesising findings from the papers. Use [N] citations inline. Be specific about methods, results, or contributions mentioned in the evidence.]

**Research Takeaway**
[Exactly ONE sentence summarising what this research direction suggests for the future.]
---

STRICT RULES:
- Use [1], [2], [3] etc. as inline citations — e.g. "Smith et al. show [1] that..."
- Do NOT invent findings not present in the evidence.
- Do NOT add a "Key Papers" section — that is added separately.
- Do NOT use any other headings.
- Total length: 200-350 words."""

    user_prompt = (
        f"Research question: {user_query}\n\n"
        f"Evidence from retrieved papers:\n{evidence_text}\n\n"
        "Write the citation-aware analysis now."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=700,
    )

    full = (resp.choices[0].message.content or "").strip()

    # Split off Research Takeaway
    if "**Research Takeaway**" in full:
        parts    = full.split("**Research Takeaway**", 1)
        analysis = parts[0].strip()
        takeaway = parts[1].strip().lstrip("\n").strip()
    else:
        analysis = full
        takeaway = ""

    return analysis, takeaway


def _build_key_papers_section(blocks: list) -> str:
    """
    Programmatically build the Key Papers section — guaranteed correct format.
    """
    lines = ["---", "", "**📄 Key Papers**", ""]

    for b in blocks:
        link_md = f"[🔗 Read on arXiv]({b['link']})" if b["link"] else ""
        lines.append(f"**[{b['index']}] {b['title']}**")
        lines.append(f"*{b['authors_full']} ({b['year']})*")
        lines.append(f"> {b['key_evidence']}")
        if link_md:
            lines.append(link_md)
        lines.append("")

    return "\n".join(lines).strip()


def _fallback_answer(user_query: str, blocks: list) -> str:
    """Clean fallback when OpenAI is unavailable."""
    lines = [
        f"**Overview**",
        f"Here are the most relevant recent papers on **{user_query}** "
        f"retrieved from arXiv ({len(blocks)} paper(s)):",
        "",
    ]
    lines.append(_build_key_papers_section(blocks))
    return "\n".join(lines)


def generate_answer_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Step 6 — Produce citation-aware output matching the architecture diagram."""
    blocks          = state.get("evidence_blocks", [])
    user_query      = state.get("user_query", "")
    validated_query = state.get("validated_query") or user_query

    # ── No papers retrieved ──────────────────────────────────────────────────
    if not blocks:
        try:
            client = __import__("openai").OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            resp   = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant. Answer clearly and concisely."},
                    {"role": "user",   "content": f"Explain: {user_query}"},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            answer = (resp.choices[0].message.content or "").strip()
        except Exception:
            answer = (
                f"I couldn't retrieve papers from arXiv right now for **{user_query}**. "
                "This may be a temporary rate-limit. Please try again in ~30 seconds, "
                "or switch to **Standard Research** mode."
            )

        state["final_answer"]    = answer
        state["confidence_note"] = "No papers retrieved — answer from model knowledge."
        return state

    # ── Build full citation-aware answer ────────────────────────────────────
    try:
        analysis, takeaway = _llm_analysis(validated_query, blocks)
    except Exception as e:
        print(f"[adaptive_rag] OpenAI failed: {e} — using fallback")
        state["final_answer"] = _fallback_answer(validated_query, blocks)
        state["confidence_note"] = f"OpenAI unavailable — structured fallback shown."
        return state

    # Assemble: LLM analysis + programmatic Key Papers + takeaway
    parts = [analysis, ""]

    parts.append(_build_key_papers_section(blocks))

    if takeaway:
        parts += ["", "---", "", f"**🔬 Research Takeaway**", f"> {takeaway}"]

    final = "\n".join(parts).strip()

    state["final_answer"] = final

    high = sum(1 for b in blocks if b.get("label") == "highly relevant")
    if high >= 2:
        state["confidence_note"] = "High confidence — multiple highly relevant papers cited."
    elif len(blocks) >= 2:
        state["confidence_note"] = "Moderate confidence — relevant papers found and cited."
    else:
        state["confidence_note"] = "Preliminary — limited papers; consider refining your query."

    notes = state.get("memory_notes", [])
    notes.append(f"[Step 6] Citation-aware answer assembled from {len(blocks)} paper(s).")
    state["memory_notes"] = notes

    return state