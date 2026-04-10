QUERY_VALIDATION_PROMPT = """
Validate the user's research query and make it retrieval-friendly without changing the meaning.
"""

RELEVANCE_GRADING_PROMPT = """
Assess whether a retrieved document is highly relevant, partially relevant, or not relevant.
"""

ANSWER_GENERATION_PROMPT = """
Generate a grounded research-style answer using only the relevant retrieved papers.
Include direct answer, analytical explanation, evidence, limitations, and sources.
"""