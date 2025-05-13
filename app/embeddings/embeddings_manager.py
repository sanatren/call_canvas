from typing import List, Dict, Any, Optional
import os

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings

from app.config.settings import EMBEDDING_MODEL, CHROMA_PERSIST_DIRECTORY

class EmbeddingsManager:
    """Manages document embeddings and vector storage."""
    
    def __init__(self):
        # Initialize the embeddings model
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Ensure persistence directory exists
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    
    def add_documents(self, doc_id: str, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            doc_id: Document ID to use as collection name
            documents: List of documents to add
        """
        # Create or get vector store for this document
        vector_store = self._get_vector_store(doc_id)
        
        # Add documents to the vector store
        vector_store.add_documents(documents)
    
    def search(self, doc_id: str, query: str, k: int = 5) -> List[Document]:
        """
        Search for documents similar to the query.
        
        Args:
            doc_id: Document ID / collection name
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        vector_store = self._get_vector_store(doc_id)
        return vector_store.similarity_search(query, k=k)
    
    def get_retriever(self, doc_id: str, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get a retriever for this document.
        
        Args:
            doc_id: Document ID / collection name
            search_kwargs: Search parameters
            
        Returns:
            Retriever object
        """
        vector_store = self._get_vector_store(doc_id)
        
        search_kwargs = search_kwargs or {"k": 4}
        return vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def _get_vector_store(self, collection_name: str) -> Chroma:
        """
        Get or create a vector store for a collection.
        
        Args:
            collection_name: Name of the collection (using doc_id)
            
        Returns:
            Chroma vector store
        """
        # Use DuckDB+Parquet backend to avoid sqlite3 version issues
        client_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            client_settings=client_settings
        )
        return vector_store 