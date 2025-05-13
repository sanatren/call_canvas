import os
import uuid
import re
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.models.document import Document as CallDocument
from app.embeddings.embeddings_manager import EmbeddingsManager
from app.utils.metadata_extractor import extract_metadata
from app.config.settings import CHUNK_SIZE, CHUNK_OVERLAP

# Import PyMuPDF but use langchain extraction for compatibility
import fitz

class DocumentProcessor:
    """Processes earnings call documents (PDF)."""
    
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
        
        # Load and process the PDF
        docs = self._load_and_split(file_path)
        
        # Extract metadata (company name, date, etc.)
        metadata = extract_metadata(file_path)
        
        # Create document record
        document = CallDocument(
            id=doc_id,
            filename=os.path.basename(file_path),
            metadata=metadata
        )
        document.save()
        
        # Generate embeddings and store vectors
        self.embeddings_manager.add_documents(doc_id, docs)
        
        return doc_id
    
    def _load_and_split(self, file_path: str) -> List[Document]:
        """
        Load PDF and split into chunks with metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of document chunks with metadata
        """
        # Use simpler LangChain extraction for better compatibility
        processed_docs = self._extract_with_langchain(file_path)
            
        # Split into chunks with metadata preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True
        )
        
        # Split while preserving metadata
        chunks = []
        for doc in processed_docs:
            # Create smaller chunks for better retrieval
            split_texts = text_splitter.split_text(doc.page_content)
            
            for i, text in enumerate(split_texts):
                # Preserve original metadata in chunks
                chunks.append(Document(
                    page_content=text,
                    metadata=doc.metadata
                ))
        
        return chunks
    
    def _extract_with_pymupdf(self, file_path: str) -> List[Document]:
        """
        Extract text from PDF using PyMuPDF (fitz) with precise positions.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of documents with precise page and line information
        """
        processed_docs = []
        
        # Open the PDF
        pdf_document = fitz.open(file_path)
        total_pages = len(pdf_document)
        
        # Process each page
        for page_num, page in enumerate(pdf_document):
            # Extract text with positioning information
            text_page = page.get_textpage()
            text_dict = page.get_text("dict")
            
            # Get blocks which represent paragraphs or sections
            blocks = text_dict.get("blocks", [])
            
            # Initialize variables for tracking
            current_speaker = ""
            current_role = ""
            page_text = ""
            line_count = 0
            
            # Process each block (paragraph)
            for block_idx, block in enumerate(blocks):
                if "lines" not in block:
                    continue
                
                # Process each line in the block
                for line_idx, line in enumerate(block["lines"]):
                    line_count += 1
                    line_text = ""
                    
                    # Combine all spans in the line
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    
                    # Skip empty lines
                    if not line_text.strip():
                        continue
                    
                    # Check for speaker info
                    speaker_info = self._extract_speaker_info(line_text)
                    if speaker_info:
                        current_speaker = speaker_info.get("speaker_name", current_speaker)
                        current_role = speaker_info.get("speaker_role", current_role)
                        if speaker_info.get("time"):
                            time_info = speaker_info.get("time")
                    
                    # Extract line y-position for better line numbering
                    y_pos = line.get("bbox", [0, 0, 0, 0])[1]
                    
                    # Create metadata for this line
                    line_metadata = {
                        "page": page_num + 1,  # 1-indexed page numbers
                        "line_number": line_count,
                        "absolute_line": f"{page_num + 1}:{line_count}",
                        "y_position": y_pos,
                        "speaker_name": current_speaker or "Narrator",
                        "speaker_role": current_role or "",
                        "block_idx": block_idx,
                        "line_idx": line_idx,
                        "total_pages": total_pages
                    }
                    
                    # Add time information if available
                    if speaker_info and speaker_info.get("time"):
                        line_metadata["time"] = speaker_info.get("time")
                    
                    # Add to page text for context
                    page_text += line_text + "\n"
                    
                    # Create document for this specific line
                    processed_docs.append(Document(
                        page_content=line_text,
                        metadata=line_metadata
                    ))
            
            # Also add the full page for broader context
            page_metadata = {
                "page": page_num + 1,
                "line_number": 1,  # First line
                "total_pages": total_pages,
                "speaker_name": "Narrator",  # Default for full page
                "is_full_page": True
            }
            
            # Add complete page content
            processed_docs.append(Document(
                page_content=page_text.strip(),
                metadata=page_metadata
            ))
        
        pdf_document.close()
        return processed_docs
    
    def _extract_with_langchain(self, file_path: str) -> List[Document]:
        """
        Extract text using LangChain's PyPDFLoader (fallback method).
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of documents with page and line information
        """
        processed_docs = []
        
        # Load PDF using LangChain
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        total_pages = len(pages)
        
        for page_idx, page_doc in enumerate(pages):
            # Get 1-indexed page number
            page_number = page_idx + 1
            
            # Process text by line
            text_content = page_doc.page_content
            lines = text_content.split('\n')
            
            # Extract speaker information and text
            current_speaker = ""
            current_role = ""
            
            # Process each line in the page
            for line_idx, line_text in enumerate(lines, 1):
                # Skip empty lines
                if not line_text.strip():
                    continue
                    
                # Check for speaker information
                speaker_info = self._extract_speaker_info(line_text)
                if speaker_info:
                    current_speaker = speaker_info.get("speaker_name", current_speaker)
                    current_role = speaker_info.get("speaker_role", current_role)
                    if speaker_info.get("time"):
                        time_info = speaker_info.get("time")
                
                # Create document for this line with metadata
                line_metadata = {
                    "page": page_number,
                    "line_number": line_idx,
                    "absolute_line": f"{page_number}:{line_idx}",
                    "speaker_name": current_speaker or "Narrator",
                    "speaker_role": current_role or "",
                    "total_pages": total_pages
                }
                
                # Add time information if available
                if speaker_info and speaker_info.get("time"):
                    line_metadata["time"] = speaker_info.get("time")
                
                processed_docs.append(Document(
                    page_content=line_text,
                    metadata=line_metadata
                ))
            
            # Also add the full page for context
            page_metadata = {
                "page": page_number,
                "line_number": 1,  # First line
                "total_pages": total_pages,
                "speaker_name": "Narrator",  # Default for full pages
                "is_full_page": True
            }
            
            processed_docs.append(Document(
                page_content=text_content,
                metadata=page_metadata
            ))
        
        return processed_docs
    
    def _extract_speaker_info(self, text: str) -> Dict[str, Any]:
        """
        Extract speaker information from text.
        
        Args:
            text: Line of text to analyze
            
        Returns:
            Dictionary with speaker info
        """
        speaker_metadata = {}
        
        # Common patterns for speaker identification
        patterns = [
            # "Name (Role):" 
            r'^([A-Za-z\s\.]+)\s*\(([^)]+)\):', 
            
            # "Name -- Role:"
            r'^([A-Za-z\s\.]+)\s*--\s*([^:]+):', 
            
            # "Name [Role]:"
            r'^([A-Za-z\s\.]+)\s*\[([^\]]+)\]:',
            
            # Time stamp with name: "[00:05:22] Name:"
            r'\[(\d{2}:\d{2}:\d{2})\]\s*([A-Za-z\s\.]+):',
            
            # Simple "Name:" pattern (if a person's name followed by colon)
            r'^([A-Za-z\s\.]{2,30}):'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) >= 2:
                    # Patterns with role information
                    if pattern == r'\[(\d{2}:\d{2}:\d{2})\]\s*([A-Za-z\s\.]+):':
                        # This is the timestamp pattern
                        speaker_metadata['time'] = match.group(1)
                        speaker_metadata['speaker_name'] = match.group(2).strip()
                    else:
                        speaker_metadata['speaker_name'] = match.group(1).strip()
                        speaker_metadata['speaker_role'] = match.group(2).strip()
                else:
                    # Patterns with just the speaker name
                    speaker_metadata['speaker_name'] = match.group(1).strip()
                    
                    # Try to infer role from common titles
                    if re.search(r'\b(CEO|CFO|COO|CTO|President|Director|Analyst|Manager|Operator)\b', text):
                        role_match = re.search(r'\b(CEO|CFO|COO|CTO|President|Director|Analyst|Manager|Operator)\b', text)
                        speaker_metadata['speaker_role'] = role_match.group(1)
                
                # Check if this contains time information
                time_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', text)
                if time_match:
                    speaker_metadata['time'] = time_match.group(1)
                
                return speaker_metadata
        
        return speaker_metadata