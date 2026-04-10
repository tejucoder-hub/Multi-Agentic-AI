
# # --------------------------------------------------------


import json
import uuid
import base64
from datetime import datetime
from pathlib import Path
import streamlit as st
import modules.research_module as research_module
import modules.data_science_module as data_science_module

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from dotenv import load_dotenv

load_dotenv()


# =================================================
# CONFIG
# =================================================
st.set_page_config(
    page_title="Multi-Agentic AI Chatbot",
    page_icon="images/Main_hub-icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.set_option("client.toolbarMode", "viewer")

PROJECT_ROOT = Path(__file__).resolve().parent
HUB_STORE_FILE = PROJECT_ROOT / "hub_chat_store.json"
RESEARCH_STORE_FILE = PROJECT_ROOT / "research_core" / "chat_store.json"
DS_PIPELINE_REPORTS = PROJECT_ROOT / "ds_core" / "pipeline_reports"
DS_PIPELINE_STORE = PROJECT_ROOT / "ds_core" / "pipeline_store"


def image_to_base64(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


research_banner_b64 = image_to_base64(str(PROJECT_ROOT / "images" / "research-button.png"))
ds_banner_b64 = image_to_base64(str(PROJECT_ROOT / "images" / "data_science-button.png"))


# =================================================
# STORE HELPERS
# =================================================
def load_hub_store():
    default_data = {
        "current_chat_id": None,
        "openai_model": "gpt-4.1-mini",
        "chats": {},
    }

    if not HUB_STORE_FILE.exists():
        return default_data

    try:
        with open(HUB_STORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "current_chat_id": data.get("current_chat_id"),
            "openai_model": data.get("openai_model", "gpt-4.1-mini"),
            "chats": data.get("chats", {}),
        }
    except Exception:
        return default_data


def save_hub_store():
    data = {
        "current_chat_id": st.session_state.hub_current_chat_id,
        "openai_model": st.session_state.hub_openai_model,
        "chats": st.session_state.hub_chats,
    }
    with open(HUB_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def make_chat_title(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "New Chat"
    return text[:42] + ("..." if len(text) > 42 else "")


def create_new_hub_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.hub_chats[chat_id] = {
        "title": "New Chat",
        "messages": [
            {
                "role": "assistant",
                "content": "Hello! I’m your Multi-Agentic AI assistant. How can I help you today?",
            }
        ],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state.hub_current_chat_id = chat_id
    save_hub_store()


def get_current_hub_chat():
    return st.session_state.hub_chats[st.session_state.hub_current_chat_id]


def delete_current_hub_chat():
    current_id = st.session_state.hub_current_chat_id
    if current_id in st.session_state.hub_chats:
        del st.session_state.hub_chats[current_id]

    if not st.session_state.hub_chats:
        create_new_hub_chat()
    else:
        st.session_state.hub_current_chat_id = list(st.session_state.hub_chats.keys())[0]
        save_hub_store()


def append_hub_note(text: str):
    return


def clean_hub_system_messages():
    unwanted = {
        "Research Agent opened from the main chatbot hub.",
        "Data Science Agent opened from the main chatbot hub.",
    }

    for chat_id, chat_data in st.session_state.hub_chats.items():
        messages = chat_data.get("messages", [])
        cleaned_messages = []

        for msg in messages:
            content = msg.get("content", "").strip()
            if content in unwanted:
                continue
            cleaned_messages.append(msg)

        st.session_state.hub_chats[chat_id]["messages"] = cleaned_messages

    save_hub_store()


# =================================================
# EXTERNAL MODULE SUMMARY
# =================================================
def get_research_store_summary():
    if not RESEARCH_STORE_FILE.exists():
        return {
            "exists": False,
            "chat_count": 0,
            "current_title": "",
            "message_count": 0,
        }

    try:
        with open(RESEARCH_STORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        chats = data.get("chats", {})
        current_chat_id = data.get("current_chat_id")
        current_chat = chats.get(current_chat_id, {}) if current_chat_id else {}
        messages = current_chat.get("messages", [])

        return {
            "exists": True,
            "chat_count": len(chats),
            "current_title": current_chat.get("title", ""),
            "message_count": len(messages),
        }
    except Exception:
        return {
            "exists": False,
            "chat_count": 0,
            "current_title": "",
            "message_count": 0,
        }


def get_data_science_summary():
    reports_count = 0
    store_count = 0

    if DS_PIPELINE_REPORTS.exists():
        reports_count = len([p for p in DS_PIPELINE_REPORTS.rglob("*") if p.is_file()])

    if DS_PIPELINE_STORE.exists():
        store_count = len([p for p in DS_PIPELINE_STORE.rglob("*") if p.is_file()])

    return {
        "exists": DS_PIPELINE_REPORTS.exists() or DS_PIPELINE_STORE.exists(),
        "reports_count": reports_count,
        "store_count": store_count,
    }


def sync_memory_and_kb():
    research_summary = get_research_store_summary()
    ds_summary = get_data_science_summary()

    st.session_state.shared_memory["last_research_summary"] = (
        f"Chats: {research_summary['chat_count']}, "
        f"Current: {research_summary['current_title'] or 'None'}, "
        f"Messages: {research_summary['message_count']}"
    )

    st.session_state.shared_memory["last_data_summary"] = (
        f"Reports: {ds_summary['reports_count']}, "
        f"Store files: {ds_summary['store_count']}"
    )

    st.session_state.knowledge_base["research_docs"] = (
        [research_summary["current_title"]] if research_summary["current_title"] else []
    )
    st.session_state.knowledge_base["dataset_info"] = (
        [f"Pipeline store files: {ds_summary['store_count']}"] if ds_summary["exists"] else []
    )
    st.session_state.knowledge_base["model_results"] = (
        [f"Pipeline reports: {ds_summary['reports_count']}"] if ds_summary["exists"] else []
    )


# =================================================
# SESSION INIT
# =================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "hub_loaded" not in st.session_state:
    data = load_hub_store()
    st.session_state.hub_chats = data["chats"]
    st.session_state.hub_current_chat_id = data["current_chat_id"]
    st.session_state.hub_openai_model = data["openai_model"]
    st.session_state.hub_loaded = True

    if not st.session_state.hub_chats:
        create_new_hub_chat()
    elif (
        not st.session_state.hub_current_chat_id
        or st.session_state.hub_current_chat_id not in st.session_state.hub_chats
    ):
        st.session_state.hub_current_chat_id = list(st.session_state.hub_chats.keys())[0]
        save_hub_store()

if "shared_memory" not in st.session_state:
    st.session_state.shared_memory = {
        "last_mode": "home",
        "last_query": "",
        "last_answer": "",
        "last_research_summary": "",
        "last_data_summary": "",
        "agent_log": [],
        "last_updated": "",
    }

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = {
        "research_docs": [],
        "dataset_info": [],
        "model_results": [],
        "notes": [],
    }

if "prefill_prompt" not in st.session_state:
    st.session_state.prefill_prompt = ""

sync_memory_and_kb()
clean_hub_system_messages()


# =================================================
# NAVIGATION
# =================================================
def go_home():
    st.session_state.page = "home"
    st.session_state.shared_memory["last_mode"] = "home"
    st.session_state.shared_memory["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sync_memory_and_kb()


def go_research():
    st.session_state.page = "research"
    st.session_state.shared_memory["last_mode"] = "research"
    st.session_state.shared_memory["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.shared_memory["agent_log"].append("Opened Research Agent")


def go_data_science():
    st.session_state.page = "data_science"
    st.session_state.shared_memory["last_mode"] = "data_science"
    st.session_state.shared_memory["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.shared_memory["agent_log"].append("Opened Data Science Agent")


# =================================================
# INTENT + GENERAL CHATBOT
# =================================================
def detect_intent(user_text: str) -> str:
    text = user_text.lower().strip()

    research_keywords = [
        "paper",
        "papers",
        "research",
        "arxiv",
        "citation",
        "citations",
        "literature",
        "survey",
        "journal",
        "pdf",
        "article",
        "references",
    ]
    data_keywords = [
        "dataset",
        "data",
        "eda",
        "csv",
        "model",
        "modelling",
        "modeling",
        "evaluation",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc",
        "churn",
        "analyse dataset",
        "analyze dataset",
    ]

    if any(word in text for word in research_keywords):
        return "research"
    if any(word in text for word in data_keywords):
        return "data_science"
    return "general"


def ask_openai(prompt: str) -> str:
    api_key = st.secrets.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "OPENAI_API_KEY is not configured. Please set it first."

    if OpenAI is None:
        return "OpenAI package is not installed."

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=st.session_state.hub_openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, professional, natural assistant inside a multi-agent AI platform. "
                        "Answer clearly and positively. "
                        "If the request is about papers, citations, arXiv, PDFs, or literature review, recommend the Research Agent. "
                        "If the request is about datasets, EDA, modelling, CSV analysis, or evaluation, recommend the Data Science Agent."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI response failed: {e}"


def ask_general_chatbot(user_text: str) -> str:
    intent = detect_intent(user_text)

    if intent == "research":
        return (
            "This looks like a research-related request. "
            "Please open the Research Agent for papers, citations, PDF summaries, and grounded research answers."
        )

    if intent == "data_science":
        return (
            "This looks like a data science task. "
            "Please open the Data Science Agent for dataset ingestion, EDA, modelling, and evaluation."
        )

    return ask_openai(user_text)


def update_memory(query: str, answer: str):
    st.session_state.shared_memory["last_mode"] = "home"
    st.session_state.shared_memory["last_query"] = query
    st.session_state.shared_memory["last_answer"] = answer
    st.session_state.shared_memory["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.shared_memory["agent_log"].append(f"Hub answered: {query[:50]}")


# =================================================
# PREMIUM PROFESSIONAL CSS
# =================================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(255,255,255,0.025), transparent 18%),
            linear-gradient(180deg, #05070b 0%, #07090d 100%);
        color: #f3f4f6;
    }}

    .block-container {{
        max-width: 1000px;
        padding-top: 4.2rem;
        padding-bottom: 1.2rem;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0b1017 0%, #090d14 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }}

    header[data-testid="stHeader"] {{
        background: rgba(15,17,21,0.82) !important;
        backdrop-filter: blur(8px);
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    .app-header {{
        padding: 0.4rem 0 0.95rem 0;
        margin-bottom: 0.25rem;
    }}

    .app-title {{
        font-size: 2.15rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.18;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
        white-space: normal;
        word-break: break-word;
    }}

    .app-subtitle {{
        color: #b9c0c9;
        font-size: 1rem;
        line-height: 1.65;
        margin-bottom: 0.25rem;
    }}

    .subtle-box {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
    }}

    .helper-text {{
        color: #c2c7cf;
        font-size: 0.97rem;
        line-height: 1.72;
    }}

    .stButton > button {{
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        background-color: #1d2127 !important;
        color: #f4f4f5 !important;
        min-height: 3rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        border: 1px solid rgba(255,255,255,0.18) !important;
        background-color: #252a31 !important;
        transform: translateY(-1px);
    }}

    .st-key-open_research_home_btn button,
    .st-key-open_ds_home_btn button {{
        min-height: 150px !important;
        border-radius: 28px !important;
        border: 1px solid rgba(90, 200, 255, 0.22) !important;
        padding: 0 !important;
        overflow: hidden !important;
        background-color: transparent !important;
        background-position: center !important;
        background-size: cover !important;
        background-repeat: no-repeat !important;
        color: transparent !important;
        font-size: 0 !important;
        line-height: 0 !important;
        text-indent: -9999px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35), 0 0 22px rgba(0, 180, 255, 0.10) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border 0.2s ease !important;
    }}

    .st-key-open_research_home_btn button:hover,
    .st-key-open_ds_home_btn button:hover {{
        transform: translateY(-2px) scale(1.01) !important;
        border: 1px solid rgba(110, 220, 255, 0.40) !important;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.42), 0 0 28px rgba(0, 190, 255, 0.18) !important;
        background-color: transparent !important;
    }}

    .st-key-open_research_home_btn button {{
        background: url("data:image/png;base64,{research_banner_b64}") center center / cover no-repeat !important;
    }}

    .st-key-open_ds_home_btn button {{
        background: url("data:image/png;base64,{ds_banner_b64}") center center / cover no-repeat !important;
    }}

    .st-key-open_research_home_btn button p,
    .st-key-open_research_home_btn button span,
    .st-key-open_ds_home_btn button p,
    .st-key-open_ds_home_btn button span {{
        display: none !important;
    }}

    .stTextInput input,
    .stTextArea textarea {{
        border-radius: 12px !important;
    }}

    div[data-testid="stChatMessage"] {{
        border-radius: 14px;
    }}

    h1, h2, h3, h4, h5, h6, p, label, div, span {{
        color: var(--text) !important;
    }}

    .sidebar-hero {{
        position: relative;
        padding: 1rem 1rem 1.05rem 1rem;
        margin-bottom: 1rem;
        border-radius: 20px;
        background:
            radial-gradient(circle at top right, rgba(76, 201, 240, 0.16), transparent 30%),
            linear-gradient(180deg, rgba(12,16,23,0.96) 0%, rgba(8,11,17,0.98) 100%);
        border: 1px solid rgba(120, 190, 255, 0.14);
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        overflow: hidden;
    }}

    .sidebar-hero-badge {{
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #9ed8ff !important;
        background: rgba(79, 172, 254, 0.10);
        border: 1px solid rgba(79, 172, 254, 0.18);
        margin-bottom: 0.7rem;
    }}

    .sidebar-status-card {{
        padding: 1rem;
        margin-bottom: 0.9rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 20px rgba(0,0,0,0.18);
    }}

    .sidebar-card-title {{
        font-size: 0.95rem;
        font-weight: 700;
        color: #eef4ff !important;
        margin-bottom: 0.85rem;
    }}

    .sidebar-stat-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.7rem;
    }}

    .sidebar-stat-box {{
        padding: 0.72rem 0.75rem;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.06);
    }}

    .sidebar-stat-label {{
        font-size: 0.74rem;
        color: #8ea0b8 !important;
        margin-bottom: 0.22rem;
    }}

    .sidebar-stat-value {{
        font-size: 0.96rem;
        font-weight: 700;
        color: #f4f8ff !important;
    }}

    .quick-prompt-note {{
        font-size: 0.84rem;
        color: #aeb8c5 !important;
        line-height: 1.5;
        margin-top: -0.2rem;
        margin-bottom: 0.8rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =================================================
# AGENT PAGES
# =================================================
if st.session_state.page == "research":
    with st.sidebar:
        st.button(
            "← Main Chatbot Hub",
            key="back_home_research_sidebar",
            on_click=go_home,
            use_container_width=True,
        )
        st.markdown("---")

    research_module.render_research_module()
    st.stop()

if st.session_state.page == "data_science":
    with st.sidebar:
        st.button(
            "← Main Chatbot Hub",
            key="back_home_ds_sidebar",
            on_click=go_home,
            use_container_width=True,
        )
        st.markdown("---")

    data_science_module.run_data_science_app()
    st.stop()


# =================================================
# HOME SIDEBAR
# =================================================
with st.sidebar:
    st.markdown("## Multi-Agentic AI")
    st.caption("Developed by Tejas Patel")

    current_mode = st.session_state.shared_memory.get("last_mode", "home").title()

    st.markdown(
        f"""
        <div class="sidebar-hero">
            <div class="sidebar-hero-badge">AI Workspace</div>
        </div>

        <div class="sidebar-status-card">
            <div class="sidebar-card-title">Workspace Overview</div>
            <div class="sidebar-stat-grid">
                <div class="sidebar-stat-box">
                    <div class="sidebar-stat-label">Mode</div>
                    <div class="sidebar-stat-value">{current_mode}</div>
                </div>
                <div class="sidebar-stat-box">
                    <div class="sidebar-stat-label">Agents</div>
                    <div class="sidebar-stat-value">2 Active</div>
                </div>
                <div class="sidebar-stat-box">
                    <div class="sidebar-stat-label">Research</div>
                    <div class="sidebar-stat-value">Ready</div>
                </div>
                <div class="sidebar-stat-box">
                    <div class="sidebar-stat-label">Data</div>
                    <div class="sidebar-stat-value">Ready</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    c1, c2 = st.columns([3, 1])
    with c1:
        if st.button("➕ New Chat", key="hub_new_chat_btn", use_container_width=True):
            create_new_hub_chat()
            st.rerun()
    with c2:
        if st.button("🗑", key="hub_delete_chat_btn", use_container_width=True):
            delete_current_hub_chat()
            st.rerun()

    st.markdown("### Chat History")
    search_text = st.text_input("Search chats", key="hub_search_chats")

    items = list(st.session_state.hub_chats.items())[::-1]
    for chat_id, chat_data in items:
        title = chat_data.get("title", "Untitled")
        if search_text and search_text.lower() not in title.lower():
            continue

        if st.button(title, key=f"hub_chat_{chat_id}", use_container_width=True):
            st.session_state.hub_current_chat_id = chat_id
            save_hub_store()
            st.rerun()

    st.markdown("---")
    st.markdown("### Try Asking")
    st.markdown(
        '<div class="quick-prompt-note">Use these quick starters to explore the platform more easily.</div>',
        unsafe_allow_html=True,
    )

    if st.button("Summarise a research topic", key="prompt_research_topic", use_container_width=True):
        st.session_state.prefill_prompt = "Summarise a research topic for me."
        st.rerun()

    if st.button("Help me analyse a dataset", key="prompt_dataset_analysis", use_container_width=True):
        st.session_state.prefill_prompt = "Help me analyse a dataset."
        st.rerun()

    if st.button("Compare machine learning models", key="prompt_compare_models", use_container_width=True):
        st.session_state.prefill_prompt = "Compare two machine learning models."
        st.rerun()

    if st.button("Guide me to the right workspace", key="prompt_workspace_help", use_container_width=True):
        st.session_state.prefill_prompt = "Help me choose between Research and Data Science workspace."
        st.rerun()

    st.markdown("---")
    st.markdown("### Shared Memory")
    st.caption(f"Last mode: {st.session_state.shared_memory.get('last_mode', 'home')}")
    st.caption(f"Last query: {st.session_state.shared_memory.get('last_query', '') or 'None'}")
    st.caption(f"Last updated: {st.session_state.shared_memory.get('last_updated', '') or 'Not yet'}")
    st.caption(f"Research summary: {st.session_state.shared_memory.get('last_research_summary', '') or 'None'}")
    st.caption(f"Data summary: {st.session_state.shared_memory.get('last_data_summary', '') or 'None'}")

    logs = st.session_state.shared_memory.get("agent_log", [])
    if logs:
        with st.expander("Open Activity Log", expanded=False):
            for log in logs[-8:][::-1]:
                st.caption(f"• {log}")
    else:
        st.caption("No activity yet.")

    st.markdown("---")
    st.markdown("### Knowledge Base")
    kb = st.session_state.knowledge_base
    st.caption(f"Research docs: {len(kb.get('research_docs', []))}")
    st.caption(f"Datasets: {len(kb.get('dataset_info', []))}")
    st.caption(f"Model results: {len(kb.get('model_results', []))}")
    st.caption(f"Notes: {len(kb.get('notes', []))}")

    with st.expander("Open Knowledge Base", expanded=False):
        if kb.get("research_docs"):
            st.markdown("**Research**")
            for item in kb["research_docs"][-5:]:
                st.caption(f"• {item}")

        if kb.get("dataset_info"):
            st.markdown("**Data Science**")
            for item in kb["dataset_info"][-5:]:
                st.caption(f"• {item}")

        if not kb.get("research_docs") and not kb.get("dataset_info") and not kb.get("model_results"):
            st.caption("Knowledge base is currently empty.")


# =================================================
# HOME PAGE
# =================================================
current_chat = get_current_hub_chat()

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">Welcome to the Multi-Agentic AI Chatbot</div>
        <div class="app-subtitle">
          A unified AI workspace for intelligent conversations, research-driven workflows, and advanced data science analysis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top1, top2 = st.columns(2)
with top1:
    if st.button("", key="open_research_home_btn", use_container_width=True):
        go_research()
        st.rerun()

with top2:
    if st.button("", key="open_ds_home_btn", use_container_width=True):
        go_data_science()
        st.rerun()

for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

chat_placeholder = st.session_state.prefill_prompt if st.session_state.prefill_prompt else "Message the main chatbot..."
user_input = st.chat_input(chat_placeholder)

if user_input:
    st.session_state.prefill_prompt = ""

    if current_chat["title"] == "New Chat":
        current_chat["title"] = make_chat_title(user_input)

    current_chat["messages"].append({"role": "user", "content": user_input})
    st.session_state.hub_chats[st.session_state.hub_current_chat_id] = current_chat
    save_hub_store()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_general_chatbot(user_input)
            st.markdown(answer)

    current_chat["messages"].append({"role": "assistant", "content": answer})
    st.session_state.hub_chats[st.session_state.hub_current_chat_id] = current_chat
    update_memory(user_input, answer)
    save_hub_store()
    st.rerun()