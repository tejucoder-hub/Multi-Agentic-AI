

# # ----------------------------------------------------------------------------------------------------



import base64
import os
import uuid
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from openai import OpenAI
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from arxiv_tool import arxiv_search

load_dotenv()

GENERATED_DIR = Path("generated_images")
GENERATED_DIR.mkdir(exist_ok=True)


class State(TypedDict):
    messages: Annotated[list, add_messages]


tools = [arxiv_search]

INITIAL_PROMPT = """
You are a production-grade academic research assistant.

PRIMARY BEHAVIOUR:
1. If the user asks for a research paper draft, write the ACTUAL paper content directly.
2. Never answer with meta text such as:
   - "your draft is ready"
   - "download the PDF below"
   - "click the link"
   - "here is the PDF"
3. The frontend handles file export. You must only return the actual written content.
4. If the user asks for recent papers, latest papers, surveys, or arXiv papers, use the arxiv_search tool.
5. When returning research papers from arXiv, use a structured format:
   - Title
   - Authors
   - Why it is relevant
   - Short summary
6. When writing a research paper draft, use a strong academic structure:
   Title:
   Abstract
   Introduction
   Background or Literature Review
   Methodology or Core Discussion
   Challenges / Limitations
   Future Directions
   Conclusion
   References
7. Keep the writing human, clear, and academically useful.
8. Do not invent download links, PDF links, or sandbox paths.
"""

PAPER_DRAFT_GUIDANCE = """
The user is asking you to WRITE a research paper draft.

Rules:
1. Return only the paper itself.
2. Do not say that the draft is ready.
3. Do not mention PDF, export, links, attachments, or downloads.
4. Start with:
   Title: <paper title>
5. Then write:
   Abstract
   Introduction
   Literature Review or Background
   Main Discussion / Methodology / Analysis
   Challenges or Limitations
   Future Directions
   Conclusion
   References
6. Make it substantial and useful, not just a short note.
"""

PAPER_SEARCH_GUIDANCE = """
The user is asking for papers or recent research.

Rules:
1. You must use the arxiv_search tool.
2. Prefer recent and relevant arXiv papers.
3. Return structured results with:
   - Title
   - Authors
   - Why it is relevant
   - Short summary
4. If the first search is broad, refine the topic conceptually and search again if needed.
5. Do not answer with generic explanation only if the user explicitly asked for papers.
"""


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return OpenAI(api_key=api_key)


def normalise_research_query(text: str) -> str:
    if not text:
        return ""

    query = f" {text.strip().lower()} "

    replacements = {
        " rag ": " retrieval augmented generation ",
        " llm ": " large language model ",
        " llms ": " large language models ",
        " cot ": " chain of thought ",
        " rlhf ": " reinforcement learning from human feedback ",
        " agentic ai ": " agentic artificial intelligence ",
        " agents ": " large language model agents ",
        " multimodal ": " multimodal large language model ",
        " reasoning ": " reasoning in large language models ",
        " hallucination ": " hallucination detection in large language models ",
        " vector db ": " vector database retrieval ",
        " mcp ": " model context protocol ",
    }

    for old, new in replacements.items():
        query = query.replace(old, new)

    return " ".join(query.split())


def looks_like_paper_search_request(text: str) -> bool:
    lowered = (text or "").lower()

    search_keywords = [
        "recent paper",
        "recent papers",
        "latest paper",
        "latest papers",
        "find papers",
        "show papers",
        "give me papers",
        "research papers",
        "survey paper",
        "survey papers",
        "arxiv",
        "latest research",
        "recent research",
        "papers on",
        "paper on",
    ]
    return any(k in lowered for k in search_keywords)


def looks_like_paper_draft_request(text: str) -> bool:
    lowered = (text or "").lower()

    draft_keywords = [
        "write a research paper",
        "write research paper",
        "draft a research paper",
        "make a research paper",
        "generate a research paper",
        "write a paper on",
        "draft a paper on",
        "create a paper on",
        "prepare a research draft",
        "research draft",
    ]
    return any(k in lowered for k in draft_keywords)


def is_image_generation_request(text: str) -> bool:
    if not text:
        return False

    keywords = [
        "generate image",
        "create image",
        "make image",
        "draw",
        "illustration",
        "diagram",
        "infographic",
        "poster",
        "visualise",
        "visualize",
        "concept art",
        "research figure",
        "workflow figure",
    ]
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def get_llm(openai_model: str = "gpt-4.1"):
    llm = ChatOpenAI(
        model=openai_model,
        temperature=0.2,
    )
    return llm.bind_tools(tools)


def build_graph(
    openai_model: str = "gpt-4.1",
    thread_id: str = "default-thread",
):
    llm = get_llm(openai_model=openai_model)
    tool_node = ToolNode(tools)

    def call_model(state: State):
        messages = state["messages"]

        full_messages = [SystemMessage(content=INITIAL_PROMPT)]

        if messages:
            last_message = messages[-1]
            last_content = getattr(last_message, "content", "")

            if isinstance(last_content, str):
                if looks_like_paper_draft_request(last_content):
                    full_messages.append(SystemMessage(content=PAPER_DRAFT_GUIDANCE))
                elif looks_like_paper_search_request(last_content):
                    refined = normalise_research_query(last_content)
                    full_messages.append(
                        SystemMessage(
                            content=f"{PAPER_SEARCH_GUIDANCE}\n\nRefined topic: {refined}"
                        )
                    )

        full_messages.extend(messages)

        response = llm.invoke(full_messages)
        return {"messages": [response]}

    def should_continue(state: State) -> Literal["tools", END]:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return graph, config


def analyse_image_with_openai(
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/png",
    model: str = "gpt-4.1",
) -> str:
    client = get_openai_client()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": data_url,
                    },
                ],
            }
        ],
    )

    return getattr(response, "output_text", "") or "No analysis returned."


def generate_image_with_openai(
    prompt: str,
    size: str = "1024x1024",
    model: str = "gpt-image-1",
) -> str:
    client = get_openai_client()

    result = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
    )

    if not result.data or not result.data[0].b64_json:
        raise ValueError("No image returned from OpenAI image generation.")

    image_bytes = base64.b64decode(result.data[0].b64_json)
    file_path = GENERATED_DIR / f"{uuid.uuid4().hex}.png"

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    return str(file_path)