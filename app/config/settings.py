import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Directory settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIRECTORY = os.path.join(BASE_DIR, "data", "uploads")
DOCUMENTS_STORE_PATH = os.path.join(BASE_DIR, "data", "documents")
CHROMA_PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data", "vectorstore")

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

# Embedding model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Good balance between quality and speed 