from typing import List, Dict, Any, Optional
import os
import logging

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from app.config.settings import (
    EMBEDDING_MODEL, 
    CHROMA_PERSIST_DIRECTORY, 
    USE_HYBRID_SEARCH,
    TOP_K_DOCUMENTS,
    RERANK_DOCUMENTS,
    COHERE_API_KEY
)

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """Manages document embeddings and vector storage with hybrid retrieval."""
    
    def __init__(self):
        # Initialize the embeddings model with explicit settings to avoid meta tensor errors
        try:
            logger.info(f"Initializing embeddings model: {EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"device": "cpu", "normalize_embeddings": True}
            )
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings with custom settings: {e}")
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
        logger.info(f"Adding {len(documents)} documents to vector store for {doc_id}")
        # Create or get vector store for this document
        vector_store = self._get_vector_store(doc_id)
        
        # Add documents to the vector store
        vector_store.add_documents(documents)
        logger.info(f"Added documents to vector store")
    
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
        retriever = self.get_retriever(doc_id, {"k": k})
        return retriever.get_relevant_documents(query)
    
    def get_retriever(self, doc_id: str, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get a retriever for this document.
        
        Args:
            doc_id: Document ID / collection name
            search_kwargs: Search parameters
            
        Returns:
            Retriever object (regular, hybrid, or reranked)
        """
        # Set default k if not provided
        search_kwargs = search_kwargs or {"k": TOP_K_DOCUMENTS}
        
        # Get vector store
        vector_store = self._get_vector_store(doc_id)
        
        # Create dense retriever
        dense_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        
        if not USE_HYBRID_SEARCH:
            return dense_retriever
        
        try:
            # Create BM25 retriever with the same documents
            logger.info("Creating hybrid retriever (BM25 + dense)")
            
            # Get docs from vector store
            docs = vector_store.get()
            if docs and "documents" in docs and docs["documents"]:
                # Create BM25 retriever
                bm25_retriever = BM25Retriever.from_documents(
                    [Document(page_content=t, metadata=m) 
                     for t, m in zip(docs["documents"], docs["metadatas"])]
                )
                bm25_retriever.k = search_kwargs.get("k", TOP_K_DOCUMENTS)
                
                # Combine retrievers - BM25 has 0.2 weight, dense has 0.8
                retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, dense_retriever],
                    weights=[0.2, 0.8]
                )
                
                # Wrap with reranker if enabled
                if RERANK_DOCUMENTS and COHERE_API_KEY:
                    try:
                        from langchain.retrievers import ContextualCompressionRetriever
                        from langchain_cohere import CohereRerank
                        
                        logger.info("Adding Cohere reranker to retriever")
                        compressor = CohereRerank(
                            model="rerank-english-v3.0", 
                            api_key=COHERE_API_KEY,
                            top_n=search_kwargs.get("k", TOP_K_DOCUMENTS)
                        )
                        return ContextualCompressionRetriever(
                            base_compressor=compressor,
                            base_retriever=retriever
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add reranker: {e}")
                
                return retriever
            else:
                logger.warning("No documents found in vector store, using dense retriever only")
                return dense_retriever
                
        except Exception as e:
            logger.warning(f"Error creating hybrid retriever: {e}, falling back to dense")
            return dense_retriever
    
    def _get_vector_store(self, collection_name: str) -> Chroma:
        """
        Get or create a vector store for a collection.
        
        Args:
            collection_name: Name of the collection (using doc_id)
            
        Returns:
            Chroma vector store
        """
        # With pysqlite3 fix, we can now use persistent storage
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        return vector_store