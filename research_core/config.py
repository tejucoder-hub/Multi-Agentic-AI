import os

RAG_PROVIDER = os.getenv("RAG_PROVIDER", "openai").strip().lower()

OPENAI_CHAT_MODEL = os.getenv("RAG_OPENAI_CHAT_MODEL", "gpt-4.1")
OPENAI_EMBED_MODEL = os.getenv("RAG_OPENAI_EMBED_MODEL", "text-embedding-3-small")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("RAG_OLLAMA_CHAT_MODEL", "qwen2.5")
OLLAMA_EMBED_MODEL = os.getenv("RAG_OLLAMA_EMBED_MODEL", "nomic-embed-text")

TOP_K = int(os.getenv("RAG_TOP_K", "3"))
MAX_ITERATIONS = int(os.getenv("RAG_MAX_ITERATIONS", "3"))

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))