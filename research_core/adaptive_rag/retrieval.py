# from urllib.parse import quote_plus
# from urllib.request import urlopen
# import xml.etree.ElementTree as ET

# from .state import AdaptiveRAGState

# print("retrieval.py loaded")

# ARXIV_API_URL = "http://export.arxiv.org/api/query"


# def _safe_text(element):
#     return element.text.strip() if element is not None and element.text else ""


# def _parse_arxiv_response(xml_text: str):
#     ns = {
#         "atom": "http://www.w3.org/2005/Atom",
#         "arxiv": "http://arxiv.org/schemas/atom",
#     }

#     root = ET.fromstring(xml_text)
#     entries = root.findall("atom:entry", ns)

#     documents = []

#     for entry in entries:
#         title = _safe_text(entry.find("atom:title", ns))
#         abstract = _safe_text(entry.find("atom:summary", ns))
#         published = _safe_text(entry.find("atom:published", ns))
#         updated = _safe_text(entry.find("atom:updated", ns))
#         entry_id = _safe_text(entry.find("atom:id", ns))

#         authors = []
#         for author in entry.findall("atom:author", ns):
#             name = _safe_text(author.find("atom:name", ns))
#             if name:
#                 authors.append(name)

#         pdf_link = ""
#         primary_link = entry_id

#         for link in entry.findall("atom:link", ns):
#             href = link.attrib.get("href", "")
#             link_type = link.attrib.get("type", "")
#             title_attr = link.attrib.get("title", "")

#             if href and not primary_link:
#                 primary_link = href

#             if title_attr == "pdf" or link_type == "application/pdf":
#                 pdf_link = href

#         documents.append(
#             {
#                 "title": title,
#                 "abstract": abstract,
#                 "authors": authors,
#                 "published": published,
#                 "updated": updated,
#                 "link": primary_link,
#                 "pdf_link": pdf_link,
#                 "source": "arXiv",
#             }
#         )

#     return documents


# def _search_arxiv(query: str, max_results: int = 5):
#     encoded_query = quote_plus(query)
#     url = (
#         f"{ARXIV_API_URL}"
#         f"?search_query=all:{encoded_query}"
#         f"&start=0"
#         f"&max_results={max_results}"
#         f"&sortBy=relevance"
#         f"&sortOrder=descending"
#     )

#     with urlopen(url, timeout=20) as response:
#         xml_bytes = response.read()

#     xml_text = xml_bytes.decode("utf-8", errors="replace")
#     return _parse_arxiv_response(xml_text)


# def retrieve_documents_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
#     print("retrieve_documents_node called")

#     query = (
#         state.get("refined_query")
#         or state.get("validated_query")
#         or state.get("user_query", "")
#     ).strip()

#     state["retrieval_queries"] = [query] if query else []

#     if not query:
#         state["retrieved_documents"] = []
#         return state

#     try:
#         documents = _search_arxiv(query=query, max_results=5)
#         state["retrieved_documents"] = documents

#         notes = state.get("memory_notes", [])
#         notes.append(f"Retrieved {len(documents)} document(s) from arXiv for query: {query}")
#         state["memory_notes"] = notes

#     except Exception as e:
#         state["retrieved_documents"] = []
#         notes = state.get("memory_notes", [])
#         notes.append(f"Retrieval error: {str(e)}")
#         state["memory_notes"] = notes

#     return state



# ------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
import time
import requests
import xml.etree.ElementTree as ET

from .state import AdaptiveRAGState

print("retrieval.py loaded")

ARXIV_API_URL = "https://export.arxiv.org/api/query"

_DISCOVERY_STOPWORDS = {
    "latest", "recent", "newest", "new", "find", "show", "give", "me",
    "papers", "paper", "research", "about", "on", "survey", "surveys",
    "articles", "article", "some", "top", "best", "related", "current",
    "in", "the", "of", "and", "or", "for", "with", "what", "is", "are",
    "tell", "explain", "how", "does", "do",
}

_ABBREVIATIONS = {
    "rag":  "retrieval augmented generation",
    "llm":  "large language model",
    "llms": "large language models",
    "cot":  "chain of thought",
    "rlhf": "reinforcement learning from human feedback",
    "mcp":  "model context protocol",
    "nlp":  "natural language processing",
    "cv":   "computer vision",
    "gnn":  "graph neural network",
}


def _clean_query(raw_query: str) -> str:
    tokens = raw_query.lower().strip().split()
    tokens = [
        t.strip(".,?!")
        for t in tokens
        if t.strip(".,?!") not in _DISCOVERY_STOPWORDS and len(t.strip(".,?!")) > 1
    ]
    text = " ".join(tokens).strip()
    expanded = f" {text} "
    for abbr, full in _ABBREVIATIONS.items():
        expanded = expanded.replace(f" {abbr} ", f" {full} ")
    result = " ".join(expanded.split()).strip()
    return result or raw_query


def _safe_text(element):
    return element.text.strip() if element is not None and element.text else ""


def _parse_arxiv_xml(xml_text: str):
    ns = {
        "atom":  "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root      = ET.fromstring(xml_text)
    documents = []

    for entry in root.findall("atom:entry", ns):
        title     = _safe_text(entry.find("atom:title",     ns))
        abstract  = _safe_text(entry.find("atom:summary",   ns))
        published = _safe_text(entry.find("atom:published", ns))
        updated   = _safe_text(entry.find("atom:updated",   ns))
        entry_id  = _safe_text(entry.find("atom:id",        ns))

        authors = []
        for author in entry.findall("atom:author", ns):
            name = _safe_text(author.find("atom:name", ns))
            if name:
                authors.append(name)

        pdf_link     = ""
        primary_link = entry_id

        for link in entry.findall("atom:link", ns):
            href       = link.attrib.get("href", "")
            link_type  = link.attrib.get("type", "")
            link_title = link.attrib.get("title", "")
            if link.attrib.get("rel") == "alternate" and href:
                primary_link = href
            if (link_type == "application/pdf" or link_title == "pdf") and href:
                pdf_link = href

        documents.append({
            "title":     title,
            "abstract":  abstract,
            "authors":   authors,
            "published": published,
            "updated":   updated,
            "link":      primary_link,
            "pdf_link":  pdf_link,
            "source":    "arXiv",
        })

    return documents


def _search_arxiv(query: str, max_results: int = 8):
    clean = _clean_query(query)
    # Use + encoding manually (arXiv prefers this over %20)
    encoded = "+".join(clean.split())

    url = (
        f"{ARXIV_API_URL}"
        f"?search_query=all:{encoded}"
        f"&start=0"
        f"&max_results={max_results}"
        f"&sortBy=submittedDate"
        f"&sortOrder=descending"
    )

    headers = {"User-Agent": "research-agent/1.0 (mailto:research@example.com)"}
    delays  = [3, 6, 12]
    last_error = None

    for attempt, delay in enumerate(delays, 1):
        try:
            # verify=False fixes Windows SSL certificate issue
            response = requests.get(url, headers=headers, timeout=40, verify=False)

            if response.status_code == 429:
                wait = delay * 3
                print(f"[retrieval] Rate limited. Waiting {wait}s (attempt {attempt}/3)...")
                time.sleep(wait)
                last_error = "Rate limited by arXiv"
                continue

            if not response.ok:
                raise RuntimeError(f"arXiv returned {response.status_code}")

            return _parse_arxiv_xml(response.text)

        except requests.exceptions.Timeout:
            print(f"[retrieval] Timeout attempt {attempt}/3. Waiting {delay*2}s...")
            time.sleep(delay * 2)
            last_error = "Timeout"
            continue

        except requests.exceptions.SSLError:
            # If verify=False still fails, skip SSL entirely
            print(f"[retrieval] SSL error attempt {attempt}/3, retrying...")
            time.sleep(delay)
            last_error = "SSL Error"
            continue

        except requests.exceptions.ConnectionError as e:
            print(f"[retrieval] Connection error attempt {attempt}/3: {e}")
            time.sleep(delay * 2)
            last_error = str(e)
            continue

    raise RuntimeError(f"arXiv failed after 3 attempts. Last: {last_error}. Try again.")


def retrieve_documents_node(state: AdaptiveRAGState) -> AdaptiveRAGState:
    print("retrieve_documents_node called")

    raw_query = (
        state.get("refined_query")
        or state.get("validated_query")
        or state.get("user_query", "")
    ).strip()

    state["retrieval_queries"] = [raw_query] if raw_query else []

    if not raw_query:
        state["retrieved_documents"] = []
        return state

    try:
        documents = _search_arxiv(query=raw_query, max_results=8)
        state["retrieved_documents"] = documents
        notes = state.get("memory_notes", [])
        notes.append(f"Retrieved {len(documents)} doc(s) for: '{_clean_query(raw_query)}'")
        state["memory_notes"] = notes

    except Exception as e:
        state["retrieved_documents"] = []
        state["retrieval_error"]     = str(e)
        notes = state.get("memory_notes", [])
        notes.append(f"Retrieval error: {str(e)}")
        state["memory_notes"] = notes

    return state