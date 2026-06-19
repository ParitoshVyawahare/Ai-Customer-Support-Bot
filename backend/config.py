"""Centralized configuration loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "embed-english-light-v3.0")
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

    # RAG hyperparameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    MULTI_QUERY_N: int = int(os.getenv("MULTI_QUERY_N", "3"))

    # Confidence thresholds (Chroma uses L2 distance — lower is better)
    HIGH_CONF_THRESHOLD: float = float(os.getenv("HIGH_CONF_THRESHOLD", "1.2"))
    MEDIUM_CONF_THRESHOLD: float = float(os.getenv("MEDIUM_CONF_THRESHOLD", "1.6"))


settings = Settings()