import json
import os
from typing import Dict, Any, Optional

from app.config.settings import DOCUMENTS_STORE_PATH

class Document:
    """Model for earnings call documents."""
    
    def __init__(self, id: str, filename: str, metadata: Dict[str, Any]):
        self.id = id
        self.filename = filename
        self.metadata = metadata
    
    def save(self) -> None:
        """Save document metadata to disk."""
        # Ensure directory exists
        os.makedirs(DOCUMENTS_STORE_PATH, exist_ok=True)
        
        # Create file path
        file_path = os.path.join(DOCUMENTS_STORE_PATH, f"{self.id}.json")
        
        # Save metadata to file
        with open(file_path, 'w') as f:
            json.dump({
                'id': self.id,
                'filename': self.filename,
                'metadata': self.metadata
            }, f)
    
    @classmethod
    def get(cls, doc_id: str) -> Optional['Document']:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document object if found, None otherwise
        """
        # Create file path
        file_path = os.path.join(DOCUMENTS_STORE_PATH, f"{doc_id}.json")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return None
        
        # Load document metadata
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Create and return document object
        return cls(
            id=data['id'],
            filename=data['filename'],
            metadata=data['metadata']
        )
    
    @classmethod
    def list_all(cls) -> list['Document']:
        """
        List all documents.
        
        Returns:
            List of Document objects
        """
        documents = []
        
        # Check if directory exists
        if not os.path.exists(DOCUMENTS_STORE_PATH):
            return documents
        
        # Load all document files
        for filename in os.listdir(DOCUMENTS_STORE_PATH):
            if filename.endswith('.json'):
                file_path = os.path.join(DOCUMENTS_STORE_PATH, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Create document object
                document = cls(
                    id=data['id'],
                    filename=data['filename'],
                    metadata=data['metadata']
                )
                documents.append(document)
        
        return documents 