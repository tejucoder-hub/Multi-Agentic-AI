# Multi-Agentic AI Chatbot — MainHub

> **Design and Evaluation of a Unified Multi-Agent Architecture for Research Knowledge Discovery and Data Science Workflows**
>
> Final Year Project — Ulster University | Student: Tejas Rajeshbhai Patel (B01035879) | Supervisor: Muhammed Sihan Haroon

##  Project Overview

**MainHub** is a unified, multi-agent AI chatbot system built as a dissertation-level final year project at Ulster University. It addresses a well-identified gap in the literature: existing intelligent assistants either support *research knowledge discovery* **or** *data science workflows* — but never both, in one coherent, continuous environment.

This system solves that by implementing **three tightly coordinated agent workspaces** within a single Streamlit application:

| Workspace | Purpose |
|---|---|
| **Main ChatHub** | Central conversational hub; handles general queries and routes to agents |
| **Research Agent Workspace** | Academic knowledge discovery via Adaptive RAG + arXiv + PDF analysis |
| **Data Science Agent Workspace** | End-to-end analytical pipeline: load → clean → EDA → model → evaluate |



---

##  Academic Context

### Research Problem

Modern knowledge workers frequently need to:
1. Explore academic literature and synthesise evidence-grounded answers
2. Perform structured data analysis and machine learning workflows

These tasks are currently siloed across separate tools — literature tools, coding notebooks, ML platforms. This creates friction, reduces continuity, and weakens efficiency.



---

## System Architecture

### High-Level Overview

<img width="1536" height="849" alt="final_workflow" src="https://github.com/user-attachments/assets/e5270de2-291c-4a95-9326-7164fe5d554e" />



### Component Interaction Flow

```
User submits query
       │
       ▼
  Is workspace selected?
       │
   ┌───┴────────────────────────────────────────┐
   │ NO (Main Hub)        YES (Agent selected)   │
   ▼                      ▼                      ▼
General Chat       Research Workspace    Data Science Workspace
(OpenAI/Ollama)    (research_core/       (ds_core/
                    frontend.py)          ai_data_science_team/
                                          ds_app/app.py)
       │                  │                      │
       ▼                  ▼                      ▼
  Direct LLM        Adaptive RAG           Analytical Pipeline
  Response          Workflow               (EDA → Model → Eval)
       │                  │                      │
       └──────────────────┴──────────────────────┘
                          │
                   Response returned to
                   Main UI (unified view)
```

### Session State & Shared Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│               Streamlit Session State                   │
│                                                         │
│  hub_chats          → Chat history (all sessions)       │
│  hub_current_chat_id→ Active conversation ID            │
│  hub_openai_model   → Selected LLM model                │
│  shared_memory      → Cross-agent context:              │
│    last_mode           last_query  last_answer          │
│    last_research_summary           last_data_summary    │
│    agent_log                                            │
│  knowledge_base     → Research docs, dataset info,      │
│                        model results, notes             │
└─────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
  hub_chat_store.json          research_core/chat_store.json
  (persisted to disk)          (persisted to disk)
```

---

##  Module Deep-Dive

### 4.1 Main ChatHub Application (`app.py`)

The central control layer of the entire system. Responsibilities:

- **Page routing**: `home` → `research` → `data_science` pages via session state
- **Intent detection**: keyword-based routing (research_keywords / data_keywords)
- **General chatbot**: direct OpenAI or Ollama completion
- **Shared state synchronisation**: syncs `shared_memory` and `knowledge_base` across workspace transitions
- **Chat persistence**: JSON-backed store (`hub_chat_store.json`)
- **UI**: sidebar with workspace navigation, chat history management, model selection

```
app.py Core Functions
─────────────────────────────────────
load_hub_store()        → Reads JSON store on startup
save_hub_store()        → Persists state to disk
create_new_hub_chat()   → UUID-keyed new conversation
detect_intent(text)     → "research" | "data_science" | "general"
ask_openai(prompt)      → Calls OpenAI Chat Completions
get_research_store_summary() → Reads research workspace state
get_data_science_summary()   → Reads DS pipeline state
sync_memory_and_kb()    → Updates shared_memory + knowledge_base
```

### 4.2 Research Agent Workspace (`research_core/`)

A dedicated academic knowledge discovery environment.
<img width="1536" height="1024" alt="ChatGPT Image Apr 10, 2026, 05_37_14 PM" src="https://github.com/user-attachments/assets/3b60ed96-684b-4f53-8969-487e954b9a28" />


#### File Responsibilities

| File | Role |
|---|---|
| `frontend.py` | Streamlit UI for the research workspace (mode selection, chat, PDF upload, export) |
| `ai_researcher_2.py` | Standard research mode: LangGraph-based agent with tools (search, image gen, analysis) |
| `arxiv_tool.py` | arXiv API integration: query → XML parse → structured paper objects |
| `adaptive_rag/workflow.py` | Orchestrates the full Adaptive RAG pipeline (stateful sequential execution) |
| `adaptive_rag/state.py` | TypedDict defining the shared state across all RAG nodes |
| `adaptive_rag/node.py` | Query validation and refinement logic |
| `adaptive_rag/retrieval.py` | arXiv API calls + XML parsing for paper retrieval |
| `adaptive_rag/grading.py` | Relevance scoring (TF-style term matching + stopword removal + term expansion) |
| `adaptive_rag/memory.py` | Saves retrieval session notes to shared memory store |
| `answer_generator.py` | Generates grounded answers from relevant document context |
| `query_refiner.py` | Expands and reformulates under-specified queries |
| `relevance_grader.py` | High-level relevance grading wrapper |
| `read_pdf.py` | PDF text extraction for uploaded research documents |
| `write_pdf.py` | LaTeX → PDF rendering for exportable research outputs |
| `shared_memory.py` | Shared cross-session memory store I/O |
| `config.py` | RAG configuration (provider, model names, chunk size, TOP_K) |
| `document_loader.py` | Document chunking and loading utilities |
| `text_chunker.py` | Text segmentation (CHUNK_SIZE=600, CHUNK_OVERLAP=100) |
| `web_search_tool.py` | Supplementary web search tool integration |
| `evaluation.py` | Research response quality evaluation utilities |


#### 4.3 RAG Configuration (`config.py`)

| Parameter | Default | Purpose |
|---|---|---|
| `RAG_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_CHAT_MODEL` | `gpt-4.1` | LLM for answer generation |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `TOP_K` | `3` | Number of top documents to use |
| `CHUNK_SIZE` | `600` | Characters per text chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |

### 4.4 Data Science Agent Workspace (`ds_core/`)

A fully structured analytical environment for tabular data workflows. Built on a modular `ai_data_science_team` package.

#### Data Science Workflow


<img width="1536" height="1024" alt="ChatGPT Image Apr 10, 2026, 05_39_25 PM" src="https://github.com/user-attachments/assets/b45740e7-2310-4045-8569-5429c8b06a66" />


```
ds_core/
├── ai_data_science_team/
│   ├── agents/                    ← Specialised sub-agents
│   │   ├── data_cleaning_agent.py      Data quality & imputation
│   │   ├── data_loader_tools_agent.py  File/DB loading tools
│   │   ├── data_visualization_agent.py Plotly/chart generation
│   │   ├── data_wrangling_agent.py     Transformation & reshaping
│   │   ├── feature_engineering_agent.py Encoding, scaling, selection
│   │   ├── sql_database_agent.py       SQL query & analysis agent
│   │   └── workflow_planner_agent.py   Multi-step plan orchestration
│   │
│   ├── ds_agents/
│   │   └── eda_tools_agent.py          Exploratory Data Analysis agent
│   │
│   ├── ml_agents/
│   │   ├── h2o_ml_agent.py             H2O AutoML integration
│   │   ├── mlflow_tools_agent.py       MLflow experiment tracking
│   │   └── model_evaluation_agent.py   Model metrics & reporting
│   │
│   ├── multiagents/
│   │   ├── pandas_data_analyst.py      Pandas-based multi-agent analysis
│   │   ├── sql_data_analyst.py         SQL-driven multi-agent analysis
│   │   └── supervisor_ds_team.py       Supervisor orchestration layer
│   │
│   ├── tools/
│   │   ├── data_loader.py              File ingestion utilities
│   │   ├── eda.py                      EDA computation tools
│   │   ├── dataframe.py                DataFrame manipulation helpers
│   │   ├── h2o.py                      H2O model building tools
│   │   ├── mlflow.py                   MLflow logging & retrieval
│   │   └── sql.py                      SQL execution tools
│   │
│   ├── templates/
│   │   └── agent_templates.py          Prompt templates for all agents
│   │
│   ├── utils/
│   │   ├── pipeline.py                 Pipeline execution engine
│   │   ├── sandbox.py                  Safe code execution sandbox
│   │   ├── regex.py                    Pattern-based output parsing
│   │   ├── logging.py                  Structured logging utilities
│   │   ├── messages.py                 LangChain message utilities
│   │   ├── html.py                     HTML rendering helpers
│   │   ├── matplotlib.py               Matplotlib utilities
│   │   └── plotly.py                   Plotly rendering helpers
│   │
│   ├── parsers/
│   │   └── parsers.py                  LLM output parsing
│   │
│   ├── ds_app/
│   │   └── app.py                      Streamlit DS workspace UI
│   │
│   └── orchestration.py                Top-level DS orchestration entry
│
└── data/                          ← Sample datasets
    ├── churn_data.csv                  10,000 records — benchmark dataset
    ├── bike_sales_data.csv             Bike sales analytics dataset
    ├── bike_model_specs.csv            Bike model specification data
    ├── dirty_dataset.csv               Intentionally messy CSV for cleaning demo
    └── northwind.db                    SQLite Northwind database (SQL agent)
```



---

##  Project File Structure

```
mainhub/
│
├── app.py                         ← Main entry point (Main ChatHub)
│
├── .streamlit/
│   ├── config.toml                ← Streamlit theme & server config
│   └── secrets.toml               ← API keys (OPENAI_API_KEY, etc.) [gitignored]
│
├── modules/
│   ├── research_module.py         ← Thin launcher for research_core frontend
│   ├── data_science_module.py     ← Thin launcher for ds_core DS app
│   ├── metrics_logger.py          ← Evaluation metrics logging helper
│   └── __init__.py
│
├── research_core/
│   ├── frontend.py                ← Research workspace Streamlit UI (~1,500 lines)
│   ├── ai_researcher_2.py         ← Standard research LangGraph agent
│   ├── arxiv_tool.py              ← arXiv API search tool
│   ├── answer_generator.py        ← LLM answer generation from retrieved docs
│   ├── query_refiner.py           ← Query expansion / reformulation
│   ├── relevance_grader.py        ← Relevance grading wrapper
│   ├── read_pdf.py                ← PDF text extraction
│   ├── write_pdf.py               ← LaTeX → PDF export
│   ├── document_loader.py         ← Document loading & chunking
│   ├── text_chunker.py            ← Text chunking utility
│   ├── shared_memory.py           ← Cross-session memory I/O
│   ├── web_search_tool.py         ← Supplementary web search
│   ├── evaluation.py              ← Research quality evaluation
│   ├── config.py                  ← RAG configuration constants
│   ├── chat_store.json            ← Persisted research chat history
│   ├── shared_memory_store.json   ← Persisted memory notes
│   │
│   ├── adaptive_rag/
│   │   ├── state.py               ← AdaptiveRAGState TypedDict
│   │   ├── workflow.py            ← Pipeline orchestration (run_adaptive_rag_workflow)
│   │   ├── node.py                ← validate_query & refine_query nodes
│   │   ├── retrieval.py           ← arXiv retrieval node
│   │   ├── grading.py             ← Relevance scoring & filtering
│   │   ├── memory.py              ← Memory save node
│   │   ├── prompt.py              ← RAG prompt templates
│   │   ├── test_run.py            ← Local test runner
│   │   └── __init__.py
│   │
│   └── output/                    ← Generated research PDFs / LaTeX
│
├── ds_core/
│   ├── requirements.txt           ← DS-specific dependencies
│   ├── setup.py                   ← Package setup for ai_data_science_team
│   │
│   ├── data/
│   │   ├── churn_data.csv         ← Benchmark dataset (10,000 rows)
│   │   ├── bike_sales_data.csv    ← Sales analytics sample
│   │   ├── bike_model_specs.csv   ← Specs lookup dataset
│   │   ├── dirty_dataset.csv      ← Messy data for cleaning demos
│   │   └── northwind.db           ← Northwind SQLite database
│   │
│   └── ai_data_science_team/
│       ├── agents/                ← Core specialised sub-agents
│       ├── ds_agents/             ← EDA-specific agent
│       ├── ml_agents/             ← ML and AutoML agents
│       ├── multiagents/           ← Supervisor + Pandas/SQL analysts
│       ├── tools/                 ← Low-level tools (EDA, H2O, MLflow, SQL)
│       ├── templates/             ← All agent prompt templates
│       ├── utils/                 ← Pipeline, sandbox, logging, regex helpers
│       ├── parsers/               ← LLM output parsers
│       ├── ds_app/app.py          ← DS workspace Streamlit UI (~2,700 lines)
│       └── orchestration.py       ← DS orchestration entry point
│
├── images/                        ← UI image assets
│   ├── Main_hub-icon.png
│   ├── research-button.png
│   ├── data_science-button.png
│   ├── chat.png
│   ├── explore.png
│   └── model.png
│
├── generated_images/              ← AI-generated images from research mode
├── output/                        ← Hub-level exported outputs
├── hub_chat_store.json            ← Persisted hub chat history
└── .gitignore
```

---

## 🛠 Technology Stack

### Core Framework

| Layer | Technology | Purpose |
|---|---|---|
| UI Framework | **Streamlit ≥ 1.32** | All three workspace UIs + session management |
| Orchestration | **LangChain ≥ 1.0 + LangGraph ≥ 1.0** | Agent pipelines, LangGraph state graphs |
| LLM Backend (Cloud) | **OpenAI API** (gpt-4.1, gpt-4.1-mini, gpt-image-1) | Chat, image generation, embeddings |
| Environment | **python-dotenv** | `.env` / secrets management |

### Research Agent Stack

| Component | Technology |
|---|---|
| Academic retrieval | **arXiv Atom API** (HTTP + XML parsing) |
| PDF ingestion | **PyPDF2** |
| PDF export | **ReportLab** + LaTeX rendering |
| Standard agent | **LangGraph** state graph |
| Adaptive RAG | Custom pipeline (validate → retrieve → grade → generate → save) |
| Web search | Requests-based web search tool |

### Data Science Agent Stack

| Component | Technology |
|---|---|
| Data processing | **Pandas ≥ 2.0, NumPy ≥ 1.24** |
| Visualisation | **Plotly ≥ 5.18** |
| Classical ML | **scikit-learn ≥ 1.3** (LR, RF, GBM, SVM) |
| Gradient Boosting | **XGBoost** |
| AutoML | **H2O.ai** |
| Experiment tracking | **MLflow** |
| DB support | **SQLAlchemy** |
| Spreadsheet I/O | **openpyxl** |
| Flow visualisation | **streamlit-flow-component** |

### Infrastructure

| Aspect | Detail |
|---|---|
| Persistence | JSON files (hub_chat_store, research chat_store, shared_memory_store) |
| Code execution | Custom `sandbox.py` for safe agent-generated code execution |
| Language | Python 3.12 |
| Package management | pip (two requirements.txt: root + ds_core) |

---

##  Installation & Setup

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12 recommended |
| pip | latest |
| OpenAI API key | Required for cloud LLM mode |
| Ollama | Optional (for fully local/offline mode) |
| Git | For cloning |

### Step 1 — Clone / Extract the Project

```bash
# If using the zip:
unzip mainhub.zip
cd mainhub
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv

# Activate:
# Linux / macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3 — Install Root Dependencies

```bash
pip install -r ds_core/requirements.txt
pip install streamlit openai langchain langchain_community langchain_openai \
            langchain_experimental langgraph python-dotenv PyPDF2 reportlab \
            pandas numpy plotly scikit-learn xgboost openpyxl requests \
            streamlit-flow-component psutil
```

> **Note:** The `ds_core/requirements.txt` lists all DS-related packages. The root project uses them directly without a separate root `requirements.txt`.

### Step 4 — Install the Data Science Package

```bash
cd ds_core
pip install -e .
cd ..
```

### Step 5 — (Optional) Install and Start Ollama for Local Mode

```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5
ollama pull nomic-embed-text
ollama serve
```

---

## 🔐 Configuration

### `.streamlit/secrets.toml`

The primary API key configuration file. This file is **gitignored** and must be created manually:

```toml
# .streamlit/secrets.toml

OPENAI_API_KEY = "sk-your-openai-api-key-here"

# Optional: for Ollama local mode
OLLAMA_BASE_URL = "http://localhost:11434"
```

### `.env` (Alternative / Supplementary)

```env
OPENAI_API_KEY=sk-your-openai-api-key-here

# RAG-specific settings (all optional — defaults shown)
RAG_PROVIDER=openai
RAG_OPENAI_CHAT_MODEL=gpt-4.1
RAG_OPENAI_EMBED_MODEL=text-embedding-3-small
RAG_OLLAMA_CHAT_MODEL=qwen2.5
RAG_OLLAMA_EMBED_MODEL=nomic-embed-text
RAG_TOP_K=3
RAG_MAX_ITERATIONS=3
RAG_CHUNK_SIZE=600
RAG_CHUNK_OVERLAP=100

# Ollama local server (if using local mode)
OLLAMA_BASE_URL=http://localhost:11434
```

### Model Selection (in-app)

Both the Research workspace and Main ChatHub allow switching between:
- **OpenAI**: `gpt-4.1`, `gpt-4.1-mini`, `gpt-image-1` (image generation)
- **Ollama**: Any locally available model (e.g. `qwen2.5`, `llama3`, `mistral`)

---

##  Running the Application

### Launch the Main ChatHub

```bash
cd mainhub
streamlit run app.py
```

The app opens at: **http://localhost:8501**

### Running the Research Workspace Standalone

```bash
cd mainhub/research_core
streamlit run frontend.py
```

### Running the Data Science Workspace Standalone

```bash
cd mainhub/ds_core/ai_data_science_team/ds_app
streamlit run app.py
```

### Testing the Adaptive RAG Pipeline (CLI)

```bash
cd mainhub/research_core/adaptive_rag
python test_run.py
```

---

##  Evaluation & Results

### Research Agent — Qualitative Performance

| Evaluation Area | Observation | Score |
|---|---|---|
| Document Understanding | PDF ingestion and concise summarisation | **84%** (High) |
| Evidence Grounding | Adaptive RAG produced more grounded outputs vs standard mode | **96%** (Excellent) |
| Literature Retrieval | Retrieved topically relevant recent arXiv papers | **86%** (High) |
| Query Understanding | Academic queries correctly interpreted | **82%** (High) |
| Response Structure | Clear, academically structured outputs | **80%** (High) |
| Workflow Adaptability | Effective across standard and adaptive modes | **84%** (High) |

### Data Science Agent — Quantitative Performance (Churn Dataset)

**Dataset:** `churn_data.csv` — Bank Customer Churn Prediction  
**Source:** Kaggle (Dhakad, 2022)  
**Split:** 80:20 train-test (8,000 / 2,000)

| Category | Metric | Value |
|---|---|---|
| **Dataset** | Records | 10,000 |
| | Attributes | 14 |
| | Missing / Duplicates | 0 / 0 |
| **Setup** | Target Variable | `exited` |
| | Train/Test Split | 80:20 |
| | Train / Test Rows | 8,000 / 2,000 |
| **Results** | Accuracy | **85.65%** |
| | Precision | **85.71%** |
| | Recall | **85.65%** |
| | F1-Score | **0.8314** |

### Data Science Agent — Qualitative Performance

| Evaluation Area | Observation | Score |
|---|---|---|
| Dataset Handling | Loaded and inspected 10,000-record churn dataset | **88%** (High) |
| Data Preparation | Preprocessing, encoding, scaling for modelling | **86%** (High) |
| Workflow Execution | Complete churn workflow with 80:20 split | **84%** (High) |
| Result Presentation | Structured outputs for interpretation | **82%** (High) |

### Dual-Mode Research Comparison

```
Standard Mode (Conversational)
─────────────────────────────────────────────────────────
✓ Fast, direct responses
✓ Handles open-ended academic questions
✓ No latency from retrieval
✗ Not grounded in real sources
✗ May hallucinate citations

Adaptive RAG Mode (Evidence-Grounded)
─────────────────────────────────────────────────────────
✓ Real arXiv paper retrieval (sortBy=relevance)
✓ Relevance grading filters noise
✓ Grounded answers with citation-aware output
✓ Confidence note on retrieval quality
✓ Auto-retry with query refinement if needed
✗ Higher latency (arXiv API + LLM)
✗ Dependent on query quality
```

---

##  Research Foundations

This project draws on and implements ideas from the following key areas of literature:

### Large Language Models
Brown et al. (2020) — GPT-3 demonstrated few-shot reasoning at scale, establishing the modern LLM paradigm that underpins both agents in this system.

### Retrieval-Augmented Generation (RAG)
Lewis et al. (2020) — Original RAG paper: the foundation for the Research Agent's evidence-grounded answer generation. The Adaptive RAG in this system extends RAG with query validation, relevance grading, and auto-refinement.

### Self-RAG
Asai et al. (2024) — Self-reflective retrieval, generate, and critique. Informed the adaptive retry loop in the Adaptive RAG workflow.

### Chain-of-Thought Prompting
Wei et al. (2022) — Structured reasoning prompting, influencing agent template design.

### Multi-Agent Systems
- Li et al. (2023) — CAMEL: communicative agents for collaborative task solving
- Hong et al. (2024) — MetaGPT: meta-programming for multi-agent frameworks
- Wang et al. (2024) — Survey on LLM-based autonomous agents

### AI for Data Science
- Ma et al. (2023) — InsightPilot: LLM-empowered automated data exploration
- Guo et al. (2024) — DS-Agent: data science via case-based reasoning

### Knowledge Graphs & Memory
Ji et al. (2022) — Knowledge representation, informing the shared memory and knowledge base design.

---

##  Limitations & Future Work

### Current Limitations

| Area | Limitation |
|---|---|
| **Retrieval** | arXiv-only retrieval; no full-text indexing of papers (abstract-level only) |
| **Grading** | Rule-based term matching; no semantic/embedding-based similarity scoring |
| **LLM Dependency** | Response quality tied to underlying LLM capability |
| **Scalability** | JSON-based persistence; not suitable for multi-user production deployments |
| **Evaluation** | Research agent quality assessed qualitatively; no standardised benchmark (e.g. RAGAS) |
| **Security** | Sandbox code execution is custom; not hardened for adversarial inputs |
| **Offline** | Adaptive RAG requires internet access for arXiv API calls |

### Future Work

1. **Semantic Retrieval**: Replace term-matching grading with embedding-based similarity (e.g. FAISS, Chroma) for more robust relevance scoring
2. **Extended Sources**: Add Semantic Scholar, PubMed, CrossRef APIs alongside arXiv
3. **Full-text RAG**: Fetch and chunk full paper PDFs rather than abstracts only
4. **LangGraph Parallelism**: Upgrade to parallel node execution in the RAG pipeline for latency reduction
5. **RAGAS Evaluation**: Integrate RAGAS for standardised RAG pipeline quality measurement (faithfulness, context recall, answer relevance)
6. **Persistent Vector Store**: Replace in-memory retrieval with a persistent vector DB (e.g. Qdrant, Weaviate)
7. **User Authentication**: Add multi-user session isolation and login support
8. **Database Persistence**: Migrate JSON stores to SQLite or PostgreSQL for production-grade persistence
9. **Streaming Responses**: Enable token-streaming for real-time response rendering
10. **Agent Traceability**: Full LangSmith tracing and audit logging per interaction

---



## 📖 References

1. Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*, vol. 33, pp. 1877–1901.
2. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*, vol. 33, pp. 9459–9474.
3. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*, vol. 30, pp. 5998–6008.
4. Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
5. Asai, A., et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *ICLR 2024*.
6. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*, vol. 35, pp. 24824–24837.
7. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL-HLT*, vol. 1, pp. 4171–4186.
8. Schick, T., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *NeurIPS*, vol. 36, pp. 68539–68551.
9. Li, G., et al. (2023). CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society. *NeurIPS*, vol. 36, pp. 51991–52008.
10. Hong, S., et al. (2024). MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. *ICLR 2024*.
11. Wang, L., et al. (2024). A Survey on Large Language Model Based Autonomous Agents. *Frontiers of Computer Science*, vol. 18, no. 6.
12. Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*, pp. 1–22.
13. Ma, P., et al. (2023). InsightPilot: An LLM-Empowered Automated Data Exploration System. *EMNLP 2023*, pp. 346–352.
14. Guo, S., et al. (2024). DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning. *ICML 2024*, vol. 235, pp. 16813–16848.
15. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*, pp. 6769–6781.
16. Guu, K., et al. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML 2020*, vol. 119, pp. 3929–3938.
17. Ji, S., et al. (2022). A Survey on Knowledge Graphs. *IEEE TNNLS*, vol. 33, no. 2, pp. 494–514.
18. He, X., Zhao, K., & Chu, X. (2021). AutoML: A Survey of the State-of-the-Art. *Knowledge-Based Systems*, vol. 212, art. 106622.
19. Bzdok, D., Yeo, B. T. T., & Poldrack, R. A. (2024). Data Science Opportunities of LLMs for Neuroscience and Biomedicine. *Neuron*, vol. 112, no. 5, pp. 698–717.
20. Dhakad, S. (2022). Bank Customer Churn Prediction. Kaggle. https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
---

*README authored for dissertation submission — Ulster University Final Year Project 2025/26*
*Student: Tejas Rajeshbhai Patel (B01035879) | Supervisor: Muhammed Sihan Haroon*
