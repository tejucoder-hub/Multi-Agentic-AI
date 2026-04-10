from ddgs import DDGS
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """
    Search the public web for current information.
    Useful for non-arXiv research context, recent news, libraries, benchmarks, and updates.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"Web search failed: {e}"

    if not results:
        return "No web results found."

    lines = []
    for i, item in enumerate(results, 1):
        title = item.get("title", "No title")
        href = item.get("href", "")
        body = item.get("body", "")
        lines.append(f"[{i}] {title}\nURL: {href}\nSnippet: {body}\n")

    return "\n".join(lines)