from urllib.parse import quote_plus
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from .state import AdaptiveRAGState

print("retrieval.py loaded")

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def _safe_text(element):
    return element.text.strip() if element is not None and element.text else ""


def _parse_arxiv_response(xml_text: str):
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", ns)

    documents = []

    for entry in entries:
        title = _safe_text(entry.find("atom:title", ns))
        abstract = _safe_text(entry.find("atom:summary", ns))
        published = _safe_text(entry.find("atom:published", ns))
        updated = _safe_text(entry.find("atom:updated", ns))
        entry_id = _safe_text(entry.find("atom:id", ns))

        authors = []
        for author in entry.findall("atom:author", ns):
            name = _safe_text(author.find("atom:name", ns))
            if name:
                authors.append(name)

        pdf_link = ""
        primary_link = entry_id

        for link in entry.findall("atom:link", ns):
            href = link.attrib.get("href", "")
            link_type = link.attrib.get("type", "")
            title_attr = link.attrib.get("title", "")

            if href and not primary_link:
                primary_link = href

            if title_attr == "pdf" or link_type == "application/pdf":
                pdf_link = href

        documents.append(
            {
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": published,
                "updated": updated,
                "link": primary_link,
                "pdf_link": pdf_link,
                "source": "arXiv",
            }
        )

    return documents


def _search_arxiv(query: str, max_results: int = 5):
    encoded_query = quote_plus(query)
    url = (
        f"{ARXIV_API_URL}"
        f"?search_query=all:{encoded_query}"
        f"&start=0"
        f"&max_results={max_results}"
        f"&sortBy=relevance"
        f"&sortOrder=descending"
    )

    with urlopen(url, timeout=20) as response:
        xml_bytes = response.read()

    xml_text = xml_bytes.decode("utf-8", errors="replace")
    return _parse_arxiv_response(xml_text)


def retrieve_documents_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    print("retrieve_documents_node called")

    query = (
        state.get("refined_query")
        or state.get("validated_query")
        or state.get("user_query", "")
    ).strip()

    state["retrieval_queries"] = [query] if query else []

    if not query:
        state["retrieved_documents"] = []
        return state

    try:
        documents = _search_arxiv(query=query, max_results=5)
        state["retrieved_documents"] = documents

        notes = state.get("memory_notes", [])
        notes.append(f"Retrieved {len(documents)} document(s) from arXiv for query: {query}")
        state["memory_notes"] = notes

    except Exception as e:
        state["retrieved_documents"] = []
        notes = state.get("memory_notes", [])
        notes.append(f"Retrieval error: {str(e)}")
        state["memory_notes"] = notes

    return state
