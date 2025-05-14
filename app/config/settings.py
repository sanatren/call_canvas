# ⚠️ SQLite Compatibility Fix ⚠️
# Swap the stdlib sqlite3 lib with the pysqlite3 package
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # fallback to built-in sqlite3

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Unstructured API settings
UNSTRUCTURED_API_KEY = os.environ.get("UNSTRUCTURED_API_KEY")
UNSTRUCTURED_API_URL = os.environ.get("UNSTRUCTURED_API_URL", "https://api.unstructured.io/general/v0/general")

# Check if running on Streamlit Cloud (it sets this environment variable)
IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE') == 'streamlit'

# Directory settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIRECTORY = os.path.join(BASE_DIR, "data", "uploads")
DOCUMENTS_STORE_PATH = "/tmp/documents" if IS_STREAMLIT_CLOUD else os.path.join(BASE_DIR, "data", "documents")
# Use /tmp directory on Streamlit Cloud for writable storage
CHROMA_PERSIST_DIRECTORY = "/tmp/vectorstore" if IS_STREAMLIT_CLOUD else os.path.join(BASE_DIR, "data", "vectorstore")

# Create directories if they don't exist
for directory in [UPLOAD_DIRECTORY, DOCUMENTS_STORE_PATH, CHROMA_PERSIST_DIRECTORY]:
    os.makedirs(directory, exist_ok=True)

# Document processing settings
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 1000

# LLM settings
# Strip any whitespace to avoid issues with the value
raw_llm_type = os.environ.get("DEFAULT_LLM_TYPE", "openai")
DEFAULT_LLM_TYPE = raw_llm_type.strip() if raw_llm_type else "openai"

# Validate the LLM type - fallback to openai if invalid
if DEFAULT_LLM_TYPE not in ["openai", "huggingface"]:
    DEFAULT_LLM_TYPE = "openai"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

# Embedding model settings - update to use better model for financial text
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Better than MiniLM for financial documents

# Retrieval settings
USE_HYBRID_SEARCH = True  # Enable hybrid (dense + sparse) search
TOP_K_DOCUMENTS = 8  # Default number of documents to retrieve
RERANK_DOCUMENTS = True  # Whether to apply reranking 
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")  # For reranking 