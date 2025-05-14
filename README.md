# CallCanvas

CallCanvas is an AI-powered tool that helps you extract insights from earnings call transcripts. Upload a transcript PDF and ask questions in natural language to get targeted information about company performance, strategy, and outlook.

![CallCanvas Screenshot](app/static/screenshot.png)

## Features

- 📊 Process and analyze earnings call transcripts with AI
- 💬 Ask natural language questions about the content
- 🔍 Get precise answers with source citations and speaker attribution
- 📱 Clean, modern UI designed for ease of use
- 🚀 Leverages a hybrid search system (dense embeddings + BM25)

## Deployment

### Running Locally

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/call_canvas.git
   cd call_canvas
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables (create a `.env` file):
   ```
   OPENAI_API_KEY=your_openai_api_key
   UNSTRUCTURED_API_KEY=your_unstructured_api_key
   DEFAULT_LLM_TYPE=openai
   ```

4. Run the app:
   ```
   streamlit run streamlit_app.py
   ```

### Deploying to Streamlit Cloud

1. Push your code to GitHub:
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/call_canvas.git
   git push -u origin main
   ```

2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account.

3. Select "New app" and choose your repository.

4. Configure the app:
   - Main file path: `streamlit_app.py`
   - Add the environment variables from above

5. Deploy and enjoy!

## Environment Variables

Set these in your `.env` file locally or in Streamlit Cloud secrets:

| Variable | Description | Required? |
|----------|-------------|-----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `UNSTRUCTURED_API_KEY` | Your Unstructured.io API key for PDF parsing | Yes |
| `DEFAULT_LLM_TYPE` | LLM provider - `openai` or `huggingface` | No (default: openai) |
| `HUGGINGFACE_API_KEY` | Your HuggingFace API key (if using HF models) | Only if `DEFAULT_LLM_TYPE=huggingface` |
| `COHERE_API_KEY` | Your Cohere API key (if using reranking) | No |

## Usage

1. Upload an earnings call transcript PDF (usually available on company investor relations websites)
2. Wait for processing (this might take a minute or two depending on the document size)
3. Ask questions about the call (e.g., "What was the revenue growth in Q3?", "What's the company's outlook for next year?")
4. Get answers with source quotes and speaker attribution
5. Use the "Clear document" button when you want to upload a new transcript

## Data Handling

All data is processed and stored temporarily. When you click "Clear document", all associated data is removed including:
- Document metadata
- Vector embeddings
- Uploaded files
- Temporary processing files

## Technical Details

CallCanvas uses:
- **Streamlit** for the user interface
- **LangChain** for orchestration
- **Unstructured.io** for high-quality PDF extraction
- **ChromaDB** for vector storage
- **OpenAI API** (or HuggingFace) for answer generation
- **BAAI/bge-base-en-v1.5** for embedding generation

The app features a hybrid search system combining dense vector embeddings with BM25 for improved retrieval.

## Debugging

If you encounter any issues:
- Check that your API keys are correctly set
- Ensure you have sufficient API credits
- For OpenAI users, check your usage limits
- For Streamlit Cloud, make sure to select a higher memory instance if processing large documents