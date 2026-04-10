# Step1: Access arXiv using URL
# import requests


# def search_arxiv_papers(topic: str, max_results: int = 5) -> dict:
#     query = "+".join(topic.lower().split())
#     for char in list('()" '):
#         if char in query:
#             print(f"Invalid character '{char}' in query: {query}")
#             raise ValueError(f"Cannot have character: '{char}' in query: {query}")
#     url = (
#             "http://export.arxiv.org/api/query"
#             f"?search_query=all:{query}"
#             f"&max_results={max_results}"
#             "&sortBy=submittedDate"
#             "&sortOrder=descending"
#         )
#     print(f"Making request to arXiv API: {url}")
#     resp = requests.get(url)
    
#     if not resp.ok:
#         print(f"ArXiv API request failed: {resp.status_code} - {resp.text}")
#         raise ValueError(f"Bad response from arXiv API: {resp}\n{resp.text}")
    
#     data = parse_arxiv_xml(resp.text)
#     return data


# # Step2: Parse XML
# import xml.etree.ElementTree as ET 
# def parse_arxiv_xml(xml_content: str) -> dict:
#     """Parse the XML content from arXiv API response."""

#     entries = []
#     ns = {
#         "atom": "http://www.w3.org/2005/Atom",
#         "arxiv": "http://arxiv.org/schemas/atom"
#     }
#     root = ET.fromstring(xml_content)
#     # Loop through each <entry> in Atom namespace
#     for entry in root.findall("atom:entry", ns):
#         # Extract authors
#         authors = [
#             author.findtext("atom:name", namespaces=ns)
#             for author in entry.findall("atom:author", ns)
#         ]
        
#         # Extract categories (term attribute)
#         categories = [
#             cat.attrib.get("term")
#             for cat in entry.findall("atom:category", ns)
#         ]
        
#         # Extract PDF link (rel="related" and type="application/pdf")
#         pdf_link = None
#         for link in entry.findall("atom:link", ns):
#             if link.attrib.get("type") == "application/pdf":
#                 pdf_link = link.attrib.get("href")
#                 break

#         entries.append({
#             "title": entry.findtext("atom:title", namespaces=ns),
#             "summary": entry.findtext("atom:summary", namespaces=ns).strip(),
#             "authors": authors,
#             "categories": categories,
#             "pdf": pdf_link
#         })

#     return {"entries": entries}



# # Step3: Convert the functionality into a tool
# from langchain_core.tools import tool


# @tool
# def arxiv_search(topic: str) -> list[dict]:
#     """Search for recently uploaded arXiv papers

#     Args:
#         topic: The topic to search for papers about

#     Returns:
#         List of papers with their metadata including title, authors, summary, etc.
#     """
#     print("ARXIV Agent called")
#     print(f"Searching arXiv for papers about: {topic}")
#     papers = search_arxiv_papers(topic)
#     if len(papers) == 0:
#         print(f"No papers found for topic: {topic}")
#         raise ValueError(f"No papers found for topic: {topic}")
#     print(f"Found {len(papers['entries'])} papers about {topic}")
#     return papers



# ---------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict
from evaluation import precision_at_k, recall_at_k
# from metrics_logger import log_metric
from langchain_core.tools import tool


ARXIV_API_URL = "http://export.arxiv.org/api/query"


def normalise_topic(topic: str) -> str:
    if not topic or not topic.strip():
        raise ValueError("Search topic cannot be empty.")

    text = f" {topic.strip().lower()} "

    replacements = {
        " rag ": " retrieval augmented generation ",
        " llm ": " large language model ",
        " llms ": " large language models ",
        " cot ": " chain of thought ",
        " rlhf ": " reinforcement learning from human feedback ",
        " mcp ": " model context protocol ",
        " agentic ai ": " agentic ai autonomous agents large language model agents ",
        " agents ": " autonomous agents large language model agents ",
        " ai agents ": " artificial intelligence agents autonomous agents ",
        " multi agent ": " multi agent systems ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[\"'():]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_query_terms(topic: str) -> List[str]:
    cleaned = normalise_topic(topic)
    stopwords = {
        "the", "a", "an", "of", "on", "for", "to", "in", "and", "or",
        "latest", "recent", "paper", "papers", "research", "about", "give", "me", "find", "show"
    }
    terms = [w for w in cleaned.split() if w not in stopwords and len(w) > 2]
    return list(dict.fromkeys(terms))


def build_arxiv_query(topic: str) -> str:
    cleaned = normalise_topic(topic)
    return "+".join(cleaned.split())


def parse_year(date_text: str) -> int | None:
    if not date_text:
        return None
    try:
        return datetime.strptime(date_text, "%Y-%m-%dT%H:%M:%SZ").year
    except ValueError:
        return None


def search_arxiv_raw(topic: str, max_results: int = 12) -> dict:
    if max_results <= 0:
        raise ValueError("max_results must be greater than 0.")

    query = build_arxiv_query(topic)

    url = (
        f"{ARXIV_API_URL}"
        f"?search_query=all:{query}"
        f"&max_results={max_results}"
        f"&sortBy=submittedDate"
        f"&sortOrder=descending"
    )

    response = requests.get(url, timeout=20)

    if not response.ok:
        raise RuntimeError(
            f"arXiv API request failed with status {response.status_code}: {response.text}"
        )

    return parse_arxiv_xml(response.text)


def parse_arxiv_xml(xml_content: str) -> dict:
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    root = ET.fromstring(xml_content)
    entries = []

    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        updated = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
        article_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()

        authors = [
            (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for author in entry.findall("atom:author", ns)
        ]
        authors = [a for a in authors if a]

        categories = [
            cat.attrib.get("term", "").strip()
            for cat in entry.findall("atom:category", ns)
            if cat.attrib.get("term")
        ]

        pdf_link = None
        article_link = article_id or None

        for link in entry.findall("atom:link", ns):
            link_type = link.attrib.get("type")
            link_title = link.attrib.get("title")
            href = link.attrib.get("href")

            if link_type == "application/pdf" and href:
                pdf_link = href

            if link.attrib.get("rel") == "alternate" and href:
                article_link = href

            if link_title == "pdf" and href:
                pdf_link = href

        year = parse_year(published)

        entries.append(
            {
                "title": title,
                "summary": summary[:1500],
                "authors": authors,
                "categories": categories,
                "published": published,
                "updated": updated,
                "year": year,
                "link": article_link,
                "pdf": pdf_link,
            }
        )

    return {"entries": entries}


def compute_relevance_score(paper: Dict, query_terms: List[str]) -> float:
    title = (paper.get("title") or "").lower()
    summary = (paper.get("summary") or "").lower()
    categories = " ".join(paper.get("categories") or []).lower()
    year = paper.get("year")

    score = 0.0

    for term in query_terms:
        if term in title:
            score += 5.0
        if term in summary:
            score += 2.0
        if term in categories:
            score += 1.5

    # phrase boosts
    joined_query = " ".join(query_terms)
    if joined_query and joined_query in title:
        score += 4.0
    if joined_query and joined_query in summary:
        score += 2.0

    # special boosts for agentic / agents queries
    agent_terms = {"agentic", "agents", "autonomous", "multi", "agent"}
    if any(t in query_terms for t in agent_terms):
        if "agent" in title or "agents" in title:
            score += 4.0
        if "agent" in summary or "agents" in summary:
            score += 2.0

    # recency bonus
    current_year = datetime.now().year
    if year:
        if year >= current_year - 1:
            score += 4.0
        elif year >= current_year - 2:
            score += 3.0
        elif year >= current_year - 3:
            score += 2.0
        elif year >= current_year - 5:
            score += 1.0

    return score


def rerank_papers(topic: str, entries: List[Dict]) -> List[Dict]:
    query_terms = extract_query_terms(topic)

    ranked = []
    for paper in entries:
        score = compute_relevance_score(paper, query_terms)
        paper_copy = dict(paper)
        paper_copy["relevance_score"] = round(score, 2)
        ranked.append(paper_copy)

    ranked.sort(
        key=lambda x: (
            x.get("relevance_score", 0),
            x.get("year") or 0,
            x.get("published") or "",
        ),
        reverse=True,
    )
    return ranked


@tool
def arxiv_search(topic: str) -> list[dict]:
    """
    Search for recent and relevant arXiv papers on a research topic.

    Args:
        topic: Research topic or keyword query.

    Returns:
        A ranked list of paper dictionaries with metadata including title, authors,
        summary, year, article link, PDF link, and relevance score.
    """
    raw = search_arxiv_raw(topic=topic, max_results=15)
    entries = raw.get("entries", [])

    if not entries:
        log_metric("total_papers_retrieved", 0, {"topic": topic})
        return [
            {
                "title": "No papers found",
                "summary": f"No recent arXiv papers were found for topic: {topic}",
                "authors": [],
                "categories": [],
                "published": "",
                "updated": "",
                "year": None,
                "link": "",
                "pdf": "",
                "relevance_score": 0,
            }
        ]

    ranked = rerank_papers(topic, entries)
    top5 = ranked[:5]

    # 1. total papers
    log_metric("total_papers_retrieved", len(top5), {"topic": topic})

    # 2. average relevance score
    avg_score = sum(p.get("relevance_score", 0) for p in top5) / len(top5)
    log_metric("average_relevance_score", round(avg_score, 2), {"topic": topic})

    # 3. precision@3 and recall@3
    retrieved_docs = [p.get("title", "") for p in top5]

    # simple demo logic:
    # relevant_docs = papers with relevance_score >= 7
    relevant_docs = [
        p.get("title", "")
        for p in top5
        if p.get("relevance_score", 0) >= 7
    ]

    p_at_3 = precision_at_k(retrieved_docs, relevant_docs, k=3)
    r_at_3 = recall_at_k(retrieved_docs, relevant_docs, k=3)

    log_metric("precision_at_3", round(p_at_3, 2), {"topic": topic})
    log_metric("recall_at_3", round(r_at_3, 2), {"topic": topic})

    return top5