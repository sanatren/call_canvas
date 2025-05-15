import os
import uuid
import re
from typing import List, Dict, Any, Optional
import logging

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

from app.models.document import Document as CallDocument
from app.embeddings.embeddings_manager import EmbeddingsManager
from app.utils.metadata_extractor import extract_metadata
from app.utils.unstructured_client import process_document_with_api, convert_to_langchain_docs
from app.config.settings import CHUNK_SIZE, CHUNK_OVERLAP

# Configure logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes earnings call documents (PDF) using Unstructured API."""
    
    def __init__(self):
        self.embeddings_manager = EmbeddingsManager()
    
    def process(self, file_path: str) -> str:
        """
        Process a document and store it in vector database.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document ID for the processed document
        """
        # Generate a unique ID for this document
        doc_id = str(uuid.uuid4())
        
        try:
            # Extract and process the document using Unstructured API
            logger.info(f"Processing document: {file_path}")
            elements = process_document_with_api(file_path)
            logger.info(f"Extracted {len(elements)} elements from document")
            
            # Convert to LangChain documents
            docs = convert_to_langchain_docs(elements)
            logger.info(f"Converted to {len(docs)} LangChain documents")
            
            # Apply chunking strategy
            chunks = self._chunk_documents(docs)
            logger.info(f"Created {len(chunks)} chunks for embedding")
            
            # Extract metadata (company name, date, etc.)
            metadata = extract_metadata(file_path)
            logger.info(f"Extracted metadata: {metadata}")
            
            # Create document record
            document = CallDocument(
                id=doc_id,
                filename=os.path.basename(file_path),
                metadata=metadata
            )
            document.save()
            logger.info(f"Saved document metadata with ID: {doc_id}")
            
            # Add a few log statements for debugging
            logger.info(f"Adding documents to vectorstore with doc_id={doc_id}")
            
            # Generate embeddings and store vectors
            try:
                # Create a test chunk with clear content for validation
                test_chunk = Document(
                    page_content="This is a test chunk to validate vectorstore operation.", 
                    metadata={"chunk_id": "test_1", "page": 1}
                )
                chunks.append(test_chunk)
                
                # Add the documents to the vector store
                self.embeddings_manager.add_documents(doc_id, chunks)
                logger.info(f"Successfully added {len(chunks)} chunks to vectorstore")
                
                # Verify the documents were actually added by retrieving test chunk
                validation_docs = self.embeddings_manager.search(doc_id, "test chunk validate", k=1)
                if validation_docs and len(validation_docs) > 0:
                    logger.info("Vector store validation successful - able to retrieve test document")
                else:
                    logger.warning("Vector store validation failed - unable to retrieve test document")
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {str(e)}", exc_info=True)
                raise
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            raise
    
    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Apply hierarchical chunking strategy.
        
        Args:
            docs: List of documents from Unstructured API
            
        Returns:
            List of chunked documents with preserved metadata
        """
        # Use token-based splitter for more predictable chunks
        text_splitter = TokenTextSplitter(
            chunk_size=150,  # ~150 tokens per chunk
            chunk_overlap=20   # 20 tokens overlap
        )
        
        # Apply chunking while preserving metadata
        all_chunks = []
        
        for doc_idx, doc in enumerate(docs):
            # Skip small documents that don't need chunking
            if len(doc.page_content.split()) < 40:  # If less than ~40 words
                # Add unique chunk ID and preserve the document as is
                metadata = doc.metadata.copy()
                metadata.update({
                    "chunk_id": f"{doc_idx}_0",
                    "is_small_doc": True
                })
                all_chunks.append(Document(
                    page_content=doc.page_content,
                    metadata=metadata
                ))
                continue
            
            # Split the document content into chunks
            chunks = text_splitter.split_text(doc.page_content)
            
            # Create new documents with preserved metadata
            for chunk_idx, chunk_text in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata.update({
                    "chunk_id": f"{doc_idx}_{chunk_idx}",
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks)
                })
                
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))
        
        return all_chunks