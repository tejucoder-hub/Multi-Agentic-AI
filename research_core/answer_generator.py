from typing import List


def generate_grounded_answer(query: str, docs: List[str], generate_text_fn) -> str:
    if not docs:
        return "I could not find any supporting research documents for this query."

    context = "\n\n".join([f"Source {i+1}: {doc}" for i, doc in enumerate(docs)])

    prompt = f"""
You are a research assistant.

Answer the user's question using only the retrieved context below.
Do not invent unsupported claims.
If the evidence is limited, say so clearly.

User question:
{query}

Retrieved context:
{context}

At the end, add:
Retrieved Evidence:
1. ...
2. ...
""".strip()

    return generate_text_fn(prompt, temperature=0.3).strip()