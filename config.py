import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Models
CHAT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# RAG settings
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
MATCH_THRESHOLD = 0.3
MATCH_COUNT = 8
EMBEDDING_BATCH_SIZE = 100
INSERT_BATCH_SIZE = 50

# Agent settings
MAX_AGENT_ITERATIONS = 3
