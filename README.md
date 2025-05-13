# CallCanvas

CallCanvas is an AI-powered tool integrated into Canvas that allows Indian equity traders to instantly extract actionable, citation-backed insights from earnings call transcripts and presentations.

## Features

- Upload earnings call PDFs
- Ask natural language questions about the content
- Get factual answers with verbatim quotes
- See speaker attribution (name and role)
- Get page and line number citations

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

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   HUGGINGFACE_API_KEY=your_huggingface_key
   DEFAULT_LLM_TYPE=openai  # or huggingface
   ```

## Running the Application

```
streamlit run app.py
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