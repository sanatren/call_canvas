from typing import Dict, Any

from app.embeddings.embeddings_manager import EmbeddingsManager

def get_retriever(doc_id: str, embeddings_manager: EmbeddingsManager, **kwargs):
    """
    Get a retriever for the specified document.
    
    Args:
        doc_id: Document ID
        embeddings_manager: Instance of EmbeddingsManager
        **kwargs: Additional parameters for retriever configuration
        
    Returns:
        Configured retriever for the document
    """
    # Default search parameters - only use parameters supported by Chroma
    search_kwargs = {
        "k": kwargs.get("k", 4)  # Number of documents to retrieve
    }
    
    # Get the retriever from the embeddings manager
    retriever = embeddings_manager.get_retriever(doc_id, search_kwargs=search_kwargs)
    
    return retriever 