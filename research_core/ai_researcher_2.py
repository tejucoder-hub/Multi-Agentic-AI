

# # # ----------------------------------------------------------------------------------------------------



# import base64
# import os
# import uuid
# from pathlib import Path
# from typing import Annotated, Literal

# from dotenv import load_dotenv
# from openai import OpenAI
# from typing_extensions import TypedDict

# from langchain_core.messages import SystemMessage
# from langchain_openai import ChatOpenAI
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import END, START, StateGraph
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode

# from arxiv_tool import arxiv_search

# load_dotenv()

# GENERATED_DIR = Path("generated_images")
# GENERATED_DIR.mkdir(exist_ok=True)


# class State(TypedDict):
#     messages: Annotated[list, add_messages]


# tools = [arxiv_search]

# INITIAL_PROMPT = """
# You are a production-grade academic research assistant.

# PRIMARY BEHAVIOUR:
# 1. If the user asks for a research paper draft, write the ACTUAL paper content directly.
# 2. Never answer with meta text such as:
#    - "your draft is ready"
#    - "download the PDF below"
#    - "click the link"
#    - "here is the PDF"
# 3. The frontend handles file export. You must only return the actual written content.
# 4. If the user asks for recent papers, latest papers, surveys, or arXiv papers, use the arxiv_search tool.
# 5. When returning research papers from arXiv, use a structured format:
#    - Title
#    - Authors
#    - Why it is relevant
#    - Short summary
# 6. When writing a research paper draft, use a strong academic structure:
#    Title:
#    Abstract
#    Introduction
#    Background or Literature Review
#    Methodology or Core Discussion
#    Challenges / Limitations
#    Future Directions
#    Conclusion
#    References
# 7. Keep the writing human, clear, and academically useful.
# 8. Do not invent download links, PDF links, or sandbox paths.
# """

# PAPER_DRAFT_GUIDANCE = """
# The user is asking you to WRITE a research paper draft.

# Rules:
# 1. Return only the paper itself.
# 2. Do not say that the draft is ready.
# 3. Do not mention PDF, export, links, attachments, or downloads.
# 4. Start with:
#    Title: <paper title>
# 5. Then write:
#    Abstract
#    Introduction
#    Literature Review or Background
#    Main Discussion / Methodology / Analysis
#    Challenges or Limitations
#    Future Directions
#    Conclusion
#    References
# 6. Make it substantial and useful, not just a short note.
# """

# PAPER_SEARCH_GUIDANCE = """
# The user is asking for papers or recent research.

# Rules:
# 1. You must use the arxiv_search tool.
# 2. Prefer recent and relevant arXiv papers.
# 3. Return structured results with:
#    - Title
#    - Authors
#    - Why it is relevant
#    - Short summary
# 4. If the first search is broad, refine the topic conceptually and search again if needed.
# 5. Do not answer with generic explanation only if the user explicitly asked for papers.
# """


# def get_openai_client() -> OpenAI:
#     api_key = os.getenv("OPENAI_API_KEY", "").strip()
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY not found in environment variables.")
#     return OpenAI(api_key=api_key)


# def normalise_research_query(text: str) -> str:
#     if not text:
#         return ""

#     query = f" {text.strip().lower()} "

#     replacements = {
#         " rag ": " retrieval augmented generation ",
#         " llm ": " large language model ",
#         " llms ": " large language models ",
#         " cot ": " chain of thought ",
#         " rlhf ": " reinforcement learning from human feedback ",
#         " agentic ai ": " agentic artificial intelligence ",
#         " agents ": " large language model agents ",
#         " multimodal ": " multimodal large language model ",
#         " reasoning ": " reasoning in large language models ",
#         " hallucination ": " hallucination detection in large language models ",
#         " vector db ": " vector database retrieval ",
#         " mcp ": " model context protocol ",
#     }

#     for old, new in replacements.items():
#         query = query.replace(old, new)

#     return " ".join(query.split())


# def looks_like_paper_search_request(text: str) -> bool:
#     lowered = (text or "").lower()

#     search_keywords = [
#         "recent paper",
#         "recent papers",
#         "latest paper",
#         "latest papers",
#         "find papers",
#         "show papers",
#         "give me papers",
#         "research papers",
#         "survey paper",
#         "survey papers",
#         "arxiv",
#         "latest research",
#         "recent research",
#         "papers on",
#         "paper on",
#     ]
#     return any(k in lowered for k in search_keywords)


# def looks_like_paper_draft_request(text: str) -> bool:
#     lowered = (text or "").lower()

#     draft_keywords = [
#         "write a research paper",
#         "write research paper",
#         "draft a research paper",
#         "make a research paper",
#         "generate a research paper",
#         "write a paper on",
#         "draft a paper on",
#         "create a paper on",
#         "prepare a research draft",
#         "research draft",
#     ]
#     return any(k in lowered for k in draft_keywords)


# def is_image_generation_request(text: str) -> bool:
#     if not text:
#         return False

#     keywords = [
#         "generate image",
#         "create image",
#         "make image",
#         "draw",
#         "illustration",
#         "diagram",
#         "infographic",
#         "poster",
#         "visualise",
#         "visualize",
#         "concept art",
#         "research figure",
#         "workflow figure",
#     ]
#     lowered = text.lower()
#     return any(k in lowered for k in keywords)


# def get_llm(openai_model: str = "gpt-4.1"):
#     llm = ChatOpenAI(
#         model=openai_model,
#         temperature=0.2,
#     )
#     return llm.bind_tools(tools)


# def build_graph(
#     openai_model: str = "gpt-4.1",
#     thread_id: str = "default-thread",
# ):
#     llm = get_llm(openai_model=openai_model)
#     tool_node = ToolNode(tools)

#     def call_model(state: State):
#         messages = state["messages"]

#         full_messages = [SystemMessage(content=INITIAL_PROMPT)]

#         if messages:
#             last_message = messages[-1]
#             last_content = getattr(last_message, "content", "")

#             if isinstance(last_content, str):
#                 if looks_like_paper_draft_request(last_content):
#                     full_messages.append(SystemMessage(content=PAPER_DRAFT_GUIDANCE))
#                 elif looks_like_paper_search_request(last_content):
#                     refined = normalise_research_query(last_content)
#                     full_messages.append(
#                         SystemMessage(
#                             content=f"{PAPER_SEARCH_GUIDANCE}\n\nRefined topic: {refined}"
#                         )
#                     )

#         full_messages.extend(messages)

#         response = llm.invoke(full_messages)
#         return {"messages": [response]}

#     def should_continue(state: State) -> Literal["tools", END]:
#         last_message = state["messages"][-1]
#         if getattr(last_message, "tool_calls", None):
#             return "tools"
#         return END

#     workflow = StateGraph(State)
#     workflow.add_node("agent", call_model)
#     workflow.add_node("tools", tool_node)

#     workflow.add_edge(START, "agent")
#     workflow.add_conditional_edges("agent", should_continue)
#     workflow.add_edge("tools", "agent")

#     checkpointer = MemorySaver()
#     graph = workflow.compile(checkpointer=checkpointer)
#     config = {"configurable": {"thread_id": thread_id}}
#     return graph, config


# def analyse_image_with_openai(
#     prompt: str,
#     image_bytes: bytes,
#     mime_type: str = "image/png",
#     model: str = "gpt-4.1",
# ) -> str:
#     client = get_openai_client()
#     b64 = base64.b64encode(image_bytes).decode("utf-8")
#     data_url = f"data:{mime_type};base64,{b64}"

#     response = client.responses.create(
#         model=model,
#         input=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "input_text", "text": prompt},
#                     {
#                         "type": "input_image",
#                         "image_url": data_url,
#                     },
#                 ],
#             }
#         ],
#     )

#     return getattr(response, "output_text", "") or "No analysis returned."


# def generate_image_with_openai(
#     prompt: str,
#     size: str = "1024x1024",
#     model: str = "gpt-image-1",
# ) -> str:
#     client = get_openai_client()

#     result = client.images.generate(
#         model=model,
#         prompt=prompt,
#         size=size,
#     )

#     if not result.data or not result.data[0].b64_json:
#         raise ValueError("No image returned from OpenAI image generation.")

#     image_bytes = base64.b64decode(result.data[0].b64_json)
#     file_path = GENERATED_DIR / f"{uuid.uuid4().hex}.png"

#     with open(file_path, "wb") as f:
#         f.write(image_bytes)

#     return str(file_path)



# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

"""
ai_researcher_2.py — Production research agent graph.

Fixes applied:
  - call_model now finds the last HUMAN/user message (not messages[-1] which
    could be an AI tool-call result in multi-turn chat) to decide which
    guidance to inject.
  - INITIAL_PROMPT rewritten to produce human-like, conversational answers
    while still being academically rigorous.
  - arxiv_search is told to receive ONLY the core topic (no discovery words).
  - Paper search guidance explicitly constructs the stripped topic and injects
    it so the LLM passes the right query to the tool.
"""

import base64
import os
import uuid
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from openai import OpenAI
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
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

# ── System prompts ───────────────────────────────────────────────────────────

INITIAL_PROMPT = """You are an expert academic research assistant — think of yourself as a brilliant PhD advisor who gives warm, clear, human-sounding answers.

CORE RULES:
1. Always respond in a natural, conversational yet academically precise tone.
   - Don't be robotic or bullet-point-heavy unless structure genuinely helps.
   - Speak like a knowledgeable colleague, not a form-filling machine.

2. For paper retrieval requests ("latest papers on X", "find research on Y", "recent LLM papers"):
   - ALWAYS call the arxiv_search tool.
   - Pass ONLY the core topic to arxiv_search — strip intent words like
     "latest", "recent", "find", "show", "papers on", "give me".
     Example: "latest LLM papers" → arxiv_search("large language model")
              "recent RAG research" → arxiv_search("retrieval augmented generation")
   - After the tool returns, present results like a colleague would:
     start with a brief framing sentence, then for each paper:
       • **Title** (year) — *Authors*
         One engaging sentence on what makes it interesting.
         [Read on arXiv](<link>)
   - If the tool returns irrelevant papers, call it again with a refined topic.

3. For paper draft requests:
   - Write the FULL paper content directly. Never say "the draft is ready" —
     just write it. The frontend handles PDF export.
   - Structure: Title → Abstract → Introduction → Literature Review →
     Methodology → Results/Analysis → Limitations → Future Work →
     Conclusion → References
   - Include at least 2 mathematical expressions where appropriate.

4. NEVER invent PDF links, download links, or sandbox paths.
5. NEVER write meta-phrases like "your draft is ready", "download below", "click here".
"""

PAPER_SEARCH_GUIDANCE = """The user wants to discover papers.

Your job:
1. Strip intent words from the query to get the core topic.
2. Call arxiv_search with ONLY the core topic string.
3. If results seem off-topic, try a more specific or broader search term.
4. Present results conversationally — not as a raw list dump.
5. Add a brief sentence saying what direction the field is moving.
"""

PAPER_DRAFT_GUIDANCE = """The user wants a full research paper draft.

Rules:
1. Output ONLY the paper content — no preamble, no "here is your draft".
2. Start immediately with: Title: <paper title>
3. Then: Abstract / Introduction / Literature Review / Methodology /
         Analysis or Discussion / Limitations / Future Directions /
         Conclusion / References
4. Make it substantial (~1500–3000 words), academically sound, and readable.
5. Include relevant mathematical notation where it adds precision.
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

# Words that express search intent but are not part of the topic
_INTENT_WORDS = {
    "latest", "recent", "newest", "find", "show", "give", "me", "papers",
    "paper", "research", "about", "on", "survey", "some", "top", "best",
    "articles", "article", "related", "current", "new",
}

_ABBREVIATIONS = {
    "rag":   "retrieval augmented generation",
    "llm":   "large language model",
    "llms":  "large language models",
    "cot":   "chain of thought",
    "rlhf":  "reinforcement learning from human feedback",
    "mcp":   "model context protocol",
    "nlp":   "natural language processing",
    "cv":    "computer vision",
    "gnn":   "graph neural network",
}


def _extract_core_topic(text: str) -> str:
    """Strip discovery-intent words and expand abbreviations."""
    tokens = text.lower().strip().split()
    tokens = [t.strip(".,?!") for t in tokens if t.strip(".,?!") not in _INTENT_WORDS]
    topic = " ".join(tokens).strip()

    expanded = f" {topic} "
    for abbr, full in _ABBREVIATIONS.items():
        expanded = expanded.replace(f" {abbr} ", f" {full} ")
    return " ".join(expanded.split()).strip() or topic


def _get_last_user_content(messages: list) -> str:
    """
    Walk the message list in reverse to find the last genuine user/human message.
    LangGraph may append AI tool-call messages, so we can't blindly take [-1].
    """
    for msg in reversed(messages):
        # LangGraph wraps human messages as HumanMessage objects
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # multi-part content (text + images)
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                return " ".join(parts).strip()
        # Also handle raw dicts (some callers pass dicts)
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def looks_like_paper_search_request(text: str) -> bool:
    lowered = (text or "").lower()
    search_keywords = [
        "recent paper", "recent papers", "latest paper", "latest papers",
        "find papers", "show papers", "give me papers", "research papers",
        "survey paper", "survey papers", "arxiv", "latest research",
        "recent research", "papers on", "paper on", "find research",
    ]
    return any(k in lowered for k in search_keywords)


def looks_like_paper_draft_request(text: str) -> bool:
    lowered = (text or "").lower()
    draft_keywords = [
        "write a research paper", "write research paper", "draft a research paper",
        "make a research paper", "generate a research paper", "write a paper on",
        "draft a paper on", "create a paper on", "prepare a research draft",
        "research draft",
    ]
    return any(k in lowered for k in draft_keywords)


def is_image_generation_request(text: str) -> bool:
    if not text:
        return False

    lowered = text.lower().strip()

    # ── Exact keyword phrases ─────────────────────────────────────────────────
    keywords = [
        # generate variants
        "generate image", "generate an image", "generate a image",
        "generate images", "generate picture", "generate a picture",
        "generate an picture", "generate photo", "generate a photo",
        "generate an photo", "generate art", "generate a art",
        "generate an art", "generate artwork", "generate a artwork",
        "generate illustration", "generate a illustration",
        "generate visual", "generate a visual", "generate graphic",
        "generate a graphic", "generate figure", "generate a figure",
        "generate diagram", "generate a diagram", "generate poster",
        "generate a poster", "generate wallpaper", "generate a wallpaper",
        "generate logo", "generate a logo", "generate icon", "generate a icon",
        "generate thumbnail", "generate a thumbnail", "generate banner",
        "generate a banner", "generate infographic", "generate a infographic",

        # create variants
        "create image", "create an image", "create a image",
        "create images", "create picture", "create a picture",
        "create an picture", "create photo", "create a photo",
        "create an photo", "create art", "create a art", "create artwork",
        "create a artwork", "create illustration", "create a illustration",
        "create visual", "create a visual", "create graphic", "create a graphic",
        "create figure", "create a figure", "create diagram", "create a diagram",
        "create poster", "create a poster", "create wallpaper", "create a wallpaper",
        "create logo", "create a logo", "create icon", "create a icon",
        "create thumbnail", "create a thumbnail", "create banner", "create a banner",
        "create infographic", "create a infographic", "create sketch",
        "create a sketch", "create painting", "create a painting",
        "create drawing", "create a drawing", "create portrait", "create a portrait",
        "create scene", "create a scene", "create background", "create a background",

        # make variants
        "make image", "make an image", "make a image",
        "make images", "make picture", "make a picture",
        "make an picture", "make photo", "make a photo",
        "make an photo", "make art", "make a art", "make artwork",
        "make a artwork", "make illustration", "make a illustration",
        "make visual", "make a visual", "make graphic", "make a graphic",
        "make figure", "make a figure", "make diagram", "make a diagram",
        "make poster", "make a poster", "make logo", "make a logo",
        "make icon", "make a icon", "make thumbnail", "make a thumbnail",
        "make banner", "make a banner", "make infographic", "make a infographic",
        "make sketch", "make a sketch", "make painting", "make a painting",
        "make drawing", "make a drawing", "make portrait", "make a portrait",
        "make scene", "make a scene", "make wallpaper", "make a wallpaper",

        # draw variants
        "draw image", "draw an image", "draw a image",
        "draw picture", "draw a picture", "draw an picture",
        "draw art", "draw a art", "draw artwork", "draw a artwork",
        "draw illustration", "draw a illustration", "draw diagram",
        "draw a diagram", "draw figure", "draw a figure",
        "draw sketch", "draw a sketch", "draw portrait", "draw a portrait",
        "draw scene", "draw a scene", "draw character", "draw a character",
        "draw me", "draw something",

        # design variants
        "design image", "design a image", "design an image",
        "design poster", "design a poster", "design logo", "design a logo",
        "design banner", "design a banner", "design graphic", "design a graphic",
        "design thumbnail", "design a thumbnail", "design icon", "design a icon",
        "design infographic", "design a infographic", "design wallpaper",
        "design a wallpaper", "design layout", "design a layout",

        # show / display variants
        "show image", "show me image", "show me an image", "show me a image",
        "show me picture", "show me a picture", "show picture",
        "display image", "display a image", "display an image",
        "display picture", "display a picture",

        # paint variants
        "paint image", "paint a image", "paint an image",
        "paint picture", "paint a picture", "paint scene", "paint a scene",
        "paint portrait", "paint a portrait",

        # render variants
        "render image", "render a image", "render an image",
        "render scene", "render a scene", "render picture", "render a picture",
        "render visual", "render a visual",

        # photo / photography
        "take a photo", "take photo", "photo of", "photograph of",
        "photorealistic", "photo realistic",

        # concept / research visuals
        "concept art", "concept image", "concept visual",
        "research figure", "research image", "research visual",
        "workflow figure", "workflow diagram", "workflow image",
        "architecture diagram", "system diagram", "flow diagram",
        "flowchart image", "mindmap", "mind map image",

        # visualise / visualize
        "visualise", "visualize", "visualisation", "visualization",

        # illustration
        "illustration of", "illustrate", "illustrated",

        # cartoon / anime / art styles
        "cartoon of", "anime style", "sketch of", "watercolor of",
        "oil painting of", "digital art", "pixel art", "3d render",
        "realistic image", "realistic photo", "realistic picture",

        # hinglish / hindi style requests
        "image bana", "image banao", "image generate karo", "image generate kar",
        "photo bana", "photo banao", "photo generate karo",
        "picture bana", "picture banao", "tasvir bana", "tasvir banao",
        "ek image", "ek photo", "ek picture",
    ]

    if any(k in lowered for k in keywords):
        return True

    # ── Pattern-based checks (broader catch) ─────────────────────────────────

    # "of X" patterns after image words
    image_nouns = ["image", "photo", "picture", "artwork", "illustration",
                   "poster", "diagram", "figure", "banner", "logo",
                   "sketch", "painting", "drawing", "portrait", "wallpaper"]

    action_verbs = ["generate", "create", "make", "draw", "design",
                    "produce", "build", "render", "paint", "show", "display"]

    for verb in action_verbs:
        for noun in image_nouns:
            if f"{verb} {noun}" in lowered:
                return True
            if f"{verb} a {noun}" in lowered:
                return True
            if f"{verb} an {noun}" in lowered:
                return True
            if f"{verb} the {noun}" in lowered:
                return True
            if f"{verb} some {noun}" in lowered:
                return True

    return False


# ── OpenAI helpers ───────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return OpenAI(api_key=api_key)


# ── Graph ────────────────────────────────────────────────────────────────────

def get_llm(openai_model: str = "gpt-4.1"):
    llm = ChatOpenAI(model=openai_model, temperature=0.3)
    return llm.bind_tools(tools)


def build_graph(openai_model: str = "gpt-4.1", thread_id: str = "default-thread"):
    llm = get_llm(openai_model=openai_model)
    tool_node = ToolNode(tools)

    def call_model(state: State):
        messages = state["messages"]

        # Always start with the base system prompt
        full_messages = [SystemMessage(content=INITIAL_PROMPT)]

        # ── Inject task-specific guidance based on the LAST USER message ──
        last_user_text = _get_last_user_content(messages)

        if last_user_text:
            if looks_like_paper_draft_request(last_user_text):
                full_messages.append(SystemMessage(content=PAPER_DRAFT_GUIDANCE))

            elif looks_like_paper_search_request(last_user_text):
                core_topic = _extract_core_topic(last_user_text)
                search_hint = (
                    f"{PAPER_SEARCH_GUIDANCE}\n\n"
                    f"Core topic extracted from user query: \"{core_topic}\"\n"
                    f"→ Call arxiv_search with this exact string: \"{core_topic}\""
                )
                full_messages.append(SystemMessage(content=search_hint))

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


# ── Multimodal helpers ───────────────────────────────────────────────────────

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
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )

    return getattr(response, "output_text", "") or "No analysis returned."


def generate_image_with_openai(
    prompt: str,
    size: str = "1024x1024",
    model: str = "gpt-image-1",
) -> str:
    client = get_openai_client()
    result = client.images.generate(model=model, prompt=prompt, size=size)

    if not result.data or not result.data[0].b64_json:
        raise ValueError("No image returned from OpenAI image generation.")

    image_bytes = base64.b64decode(result.data[0].b64_json)
    file_path = GENERATED_DIR / f"{uuid.uuid4().hex}.png"

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    return str(file_path)