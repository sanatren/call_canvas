# CallCanvas

AI-powered earnings call insights - Extract valuable information from earnings call transcripts using state-of-the-art RAG techniques.

## Overview

CallCanvas is a Streamlit application that uses advanced Retrieval-Augmented Generation (RAG) techniques to extract and analyze information from earnings call transcripts. It allows you to:

1. Upload PDF transcripts
2. Ask natural language questions
3. Get concise, cited answers with page references
4. View the original quotes from the transcript

## Architecture

CallCanvas implements a robust, modern RAG pipeline:

### 1. Document Processing
- Uses [Unstructured](https://unstructured.io) with hi-res strategy to extract structured elements (paragraphs, tables, etc.)
- Preserves page numbers, speaker information, and section structure
- Intelligently groups content by speaker to maintain context

### 2. Hierarchical Chunking
- Primary splitting by logical blocks (speaker turns, tables, etc.)
- Secondary splitting by tokens (150 tokens, 20-token overlap)
- Preserves metadata across chunks for accurate citation

### 3. Hybrid Retrieval
- [BGE-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) dense embeddings (optimized for financial text)
- BM25 sparse retrieval
- Linear combination of both scores (80% dense, 20% sparse)
- Optional reranking with Cohere Rerank

### 4. LLM Answer Generation
- Supports OpenAI and Hugging Face models
- Structured prompting for citations
- Chunk IDs for precise source identification

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with API keys:
   ```
   OPENAI_API_KEY=your-openai-api-key
   UNSTRUCTURED_API_KEY=your-unstructured-api-key
   COHERE_API_KEY=your-cohere-api-key  # Optional for reranking
   ```
4. Run the application:
   ```
   streamlit run streamlit_app.py
   ```

## Features

- **Hi-fidelity Parsing**: Accurately extracts text, tables, and structure from PDF transcripts
- **Accurate Citations**: Points to exact pages and speakers in the transcript
- **Hybrid Search**: Combines keyword and semantic search for better retrieval
- **Optimized for Earnings Calls**: Specialized for financial transcript formats and terminology

## Requirements

- Python 3.8+
- API keys for:
  - OpenAI or HuggingFace (for LLM)
  - Unstructured.io (for document processing)
  - Cohere (optional, for reranking)

## Deployment

CallCanvas can be deployed on Streamlit Cloud or any platform that supports Streamlit applications.

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── app/
│   ├── core/               # Core application logic
│   │   ├── document_processor.py  # PDF processing
│   │   └── query_engine.py        # Question answering
│   ├── models/             # LLM models and document schemas
│   │   ├── document.py     # Document model
│   │   └── llm.py          # LLM configuration
│   ├── embeddings/         # Vector embeddings
│   │   └── embeddings_manager.py  # Embedding generation and storage
│   ├── retrieval/          # Document retrieval
│   │   └── retriever.py    # Retrieval logic
│   ├── utils/              # Utility functions
│   │   ├── file_utils.py   # File handling utilities
│   │   └── metadata_extractor.py  # Extract metadata from PDFs
│   ├── components/         # UI components
│   │   ├── results.py      # Results display
│   │   └── sidebar.py      # Sidebar UI
│   └── config/             # Configuration
│       └── settings.py     # App settings
├── data/                   # Data storage (created at runtime)
└── requirements.txt        # Project dependencies
```

## Usage

1. Upload an earnings call PDF through the interface
2. Ask questions about the earnings call in natural language
3. View the results with citations and speaker attribution

## Extending the Project

- Add support for more document types (PPTX, DOCX)
- Implement advanced speaker detection
- Add sentiment analysis
- Support multiple concurrent documents
- Enable document comparison