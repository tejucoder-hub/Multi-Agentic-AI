from typing import List


def grade_relevance(query: str, docs: List[str], generate_text_fn) -> int:
    if not docs:
        return 0

    docs_text = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])

    prompt = f"""
You are a retrieval evaluator.

Query:
{query}

Retrieved documents:
{docs_text}

Rate the overall relevance from 1 to 10.
Return only one integer.
""".strip()

    result = generate_text_fn(prompt, temperature=0)

    try:
        score = int(result.strip())
        return max(1, min(score, 10))
    except Exception:
        digits = "".join(ch for ch in result if ch.isdigit())
        if digits:
            try:
                score = int(digits[:2])
                return max(1, min(score, 10))
            except Exception:
                pass
        return 5