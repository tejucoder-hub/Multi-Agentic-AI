def refine_query(query: str, generate_text_fn) -> str:
    prompt = f"""
Rewrite this user query for better academic retrieval.
Keep the meaning the same, but improve clarity and retrieval quality.

Original query:
{query}

Return only the improved query.
""".strip()

    refined = generate_text_fn(prompt, temperature=0.2).strip()
    return refined if refined else query