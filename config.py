"""Configuration constants for the Council of Experts system."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "corpus"
AUDIT_LOG_DIR = BASE_DIR / "audit_logs"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
CORPUS_DIR.mkdir(exist_ok=True)
AUDIT_LOG_DIR.mkdir(exist_ok=True)

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB collection
CHROMA_COLLECTION = "governance_corpus"

# Chunking parameters
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

# RAG retrieval
TOP_K_CHUNKS = 5

# LLM configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "groq"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.3

# Document domain tags — maps filename substrings to agent domains
DOMAIN_TAGS = {
    "risk": ["nist", "iso", "rmf", "42001", "owasp", "mitre", "attack", "threat", "vulnerability", "robustness", "reliability"],
    "safety": ["safety", "harm", "bias", "fairness", "privacy", "deception", "accountability", "transparency", "red_team", "redteam"],
    "governance": ["eu_ai_act", "ai_act", "regulation", "governing", "compliance", "unesco", "oversight", "human_rights"],
}

# Agent names
AGENT_RISK = "Risk & Technical Standards Expert"
AGENT_SAFETY = "AI Safety & Red-Team Expert"
AGENT_GOVERNANCE = "Governance & Regulatory Expert"
