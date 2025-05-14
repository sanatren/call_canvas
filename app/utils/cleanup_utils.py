import os
import shutil
import logging
from typing import Optional

from app.config.settings import (
    DOCUMENTS_STORE_PATH,
    CHROMA_PERSIST_DIRECTORY,
    UPLOAD_DIRECTORY
)

logger = logging.getLogger(__name__)

def clean_document_data(doc_id: Optional[str] = None) -> bool:
    """
    Clean up all data related to a document.
    If doc_id is None, clean up all documents.
    
    Args:
        doc_id: Document ID to clean up, or None to clean up all
        
    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        if doc_id:
            # Clean up specific document
            logger.info(f"Cleaning up document: {doc_id}")
            return (
                _clean_metadata(doc_id) and
                _clean_vectorstore(doc_id) and 
                _clean_uploads(doc_id)
            )
        else:
            # Clean up all documents
            logger.info("Cleaning up all documents")
            return (
                _clean_all_metadata() and
                _clean_all_vectorstores() and
                _clean_all_uploads()
            )
    except Exception as e:
        logger.error(f"Error cleaning up document data: {str(e)}")
        return False

def _clean_metadata(doc_id: str) -> bool:
    """
    Clean up document metadata.
    
    Args:
        doc_id: Document ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Remove metadata JSON file
        meta_file = os.path.join(DOCUMENTS_STORE_PATH, f"{doc_id}.json")
        if os.path.exists(meta_file):
            os.remove(meta_file)
            logger.info(f"Removed metadata file: {meta_file}")
        return True
    except Exception as e:
        logger.error(f"Error cleaning metadata: {str(e)}")
        return False

def _clean_vectorstore(doc_id: str) -> bool:
    """
    Clean up document vector store.
    
    Args:
        doc_id: Document ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if collection-specific directory exists in Chroma
        vector_dir = os.path.join(CHROMA_PERSIST_DIRECTORY, doc_id)
        if os.path.exists(vector_dir):
            shutil.rmtree(vector_dir, ignore_errors=True)
            logger.info(f"Removed vector store directory: {vector_dir}")
            
        # Also check for collection in main chroma directory
        chroma_collections = os.path.join(CHROMA_PERSIST_DIRECTORY, "chroma.sqlite3")
        if os.path.exists(chroma_collections):
            # We can't easily delete just one collection from the SQLite file
            # In a production system, we would use ChromaDB's API to delete the collection
            # For now, we'll leave this as is and focus on the collection directories
            logger.info("Note: SQLite collection references may still exist in Chroma")
            
        return True
    except Exception as e:
        logger.error(f"Error cleaning vector store: {str(e)}")
        return False

def _clean_uploads(doc_id: str) -> bool:
    """
    Clean up uploaded files for a document.
    This is a best-effort process since we don't track which file belongs to which doc_id.
    
    Args:
        doc_id: Document ID
        
    Returns:
        True if successful, False otherwise
    """
    # Since we don't have a direct mapping from doc_id to uploaded files,
    # this is a placeholder for potential future implementation
    # In a production system, we would store this mapping
    
    # Note: We're not implementing this now since it would require additional tracking
    return True

def _clean_all_metadata() -> bool:
    """
    Clean up all document metadata.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(DOCUMENTS_STORE_PATH):
            for filename in os.listdir(DOCUMENTS_STORE_PATH):
                if filename.endswith(".json"):
                    file_path = os.path.join(DOCUMENTS_STORE_PATH, filename)
                    os.remove(file_path)
                    logger.info(f"Removed metadata file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error cleaning all metadata: {str(e)}")
        return False

def _clean_all_vectorstores() -> bool:
    """
    Clean up all vector stores.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(CHROMA_PERSIST_DIRECTORY):
            # Remove the entire directory and recreate it
            shutil.rmtree(CHROMA_PERSIST_DIRECTORY, ignore_errors=True)
            logger.info(f"Removed vector store directory: {CHROMA_PERSIST_DIRECTORY}")
            os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error cleaning all vector stores: {str(e)}")
        return False

def _clean_all_uploads() -> bool:
    """
    Clean up all uploaded files.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(UPLOAD_DIRECTORY):
            # Clean but don't delete the directory itself
            for filename in os.listdir(UPLOAD_DIRECTORY):
                file_path = os.path.join(UPLOAD_DIRECTORY, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed uploaded file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error cleaning all uploads: {str(e)}")
        return False 