import os
import requests
import tempfile
import re
from typing import List, Dict, Any
import logging
from pathlib import Path
from typing import Dict
from app.config.settings import UNSTRUCTURED_API_KEY, UNSTRUCTURED_API_URL

logger = logging.getLogger(__name__)

def process_document_with_api(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a document using the Unstructured API with hi_res strategy.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of extracted elements with metadata
    """
    if not UNSTRUCTURED_API_KEY:
        raise ValueError("UNSTRUCTURED_API_KEY not set in environment variables")
    
    logger.info(f"Processing document with Unstructured API: {file_path}")
    
    headers = {
        "unstructured-api-key": UNSTRUCTURED_API_KEY,
        "accept": "application/json"
    }
    
    with open(file_path, "rb") as f:
        files = {"files": (Path(file_path).name, f, "application/pdf")}
        
        # Parameters for hi_res strategy
        data = {
            "strategy": "hi_res",
            "coordinates": "true",  # Get coordinate information
            "languages": "eng"       # English language
        }
        
        try:
            response = requests.post(
                UNSTRUCTURED_API_URL,
                headers=headers,
                files=files,
                data=data
            )
            
            response.raise_for_status()
            elements = response.json()
            logger.info(f"Extracted {len(elements)} elements from document")
            return elements
            
        except requests.RequestException as e:
            logger.error(f"Error calling Unstructured API: {str(e)}")
            raise Exception(f"Error processing document with Unstructured API: {str(e)}")
        
# ▶︎ NEW – one compiled regex, anchored to the *start* of the element
_SPEAKER_LINE_RE = re.compile(
    r"""
    ^\s*
    (?:\[(?P<time>\d{1,2}:\d{2}:\d{2})\]\s*)?           # optional [hh:mm:ss]
    (?P<name>                                           # speaker name OR tag
        (?:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})    # up to 3 capitalised tokens
        |Operator|Analyst|Participant
    )
    \s*
    (?:                                                 # optional “- Role” or “(Role)”
        (?:[–\-—]|--)\s*(?P<role_dash>[A-Za-z/& ]{2,40})|
        \((?P<role_paren>[^)]+)\)
    )?
    \s*:
    """,
    re.VERBOSE,
)


def extract_speaker_info(text: str) -> dict[str, str]:
    """
    Return {'speaker_name': ..., 'speaker_role': ..., 'time': ...} if the first
    line of `text` starts with a speaker tag like any of these:

        Krishna Kanumuri: …
        Krishna Kanumuri – Executive Director: …
        [00:02:15] Operator: …
        John Doe (CFO):
        Analyst: …

    Otherwise returns an empty dict.
    """
    first_line = text.splitlines()[0][:120]  # inspect only the beginning
    m = _SPEAKER_LINE_RE.match(first_line)
    if not m:
        return {}

    info: dict[str, str] = {"speaker_name": m["name"]}
    role = m["role_dash"] or m["role_paren"]
    if role:
        info["speaker_role"] = role.strip()
    if m["time"]:
        info["time"] = m["time"]
    return info

def convert_to_langchain_docs(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Unstructured API elements to LangChain documents.
    
    Args:
        elements: List of elements from Unstructured API
        
    Returns:
        List of consolidated elements for further processing
    """
    from langchain.schema import Document
    
    processed_docs = []
    
    # First, group elements by page number
    pages = {}
    for element in elements:
        page_num = element.get("metadata", {}).get("page_number", 1)
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(element)
    
    # Process elements page by page
    for page_num, page_elements in pages.items():
        current_speaker = "Narrator"
        current_role = ""
        current_timestamp = ""
        speaker_buf = []
        meta_buf = None
        
        # Process narrative text elements, grouping by speaker
        for element in page_elements:
            element_type = element.get("type")
            # Skip certain element types that aren't useful for RAG
            if element_type in ["UncategorizedText", "Footer", "Header", "Title"]:
                continue
                
            text = element.get("text", "").strip()
            if not text:
                continue
                
            # Add coordinates and metadata
            metadata = {
                "page": page_num, 
                "element_type": element_type,
                "total_pages": max(pages.keys()),
                "speaker_name": current_speaker,
                "speaker_role": current_role,
                "time": current_timestamp
            }
            
            # Add coordinates if available
            if "metadata" in element and "coordinates" in element["metadata"]:
                coords = element["metadata"]["coordinates"]
                metadata.update({
                    "x1": coords.get("x1", 0),
                    "y1": coords.get("y1", 0),
                    "x2": coords.get("x2", 0),
                    "y2": coords.get("y2", 0),
                })
            
            # Check for speaker in narrative text
            if element_type == "NarrativeText":
                speaker_info = extract_speaker_info(text)
                
                if speaker_info:
                    # Start of a new speaker - flush previous buffer
                    if speaker_buf and meta_buf:
                        doc = Document(
                            page_content="\n".join(speaker_buf),
                            metadata=meta_buf
                        )
                        processed_docs.append(doc)
                        speaker_buf = []
                    
                    # Update speaker metadata
                    current_speaker = speaker_info.get("speaker_name", current_speaker)
                    current_role = speaker_info.get("speaker_role", current_role)
                    if "time" in speaker_info:
                        current_timestamp = speaker_info.get("time")
                    
                    metadata.update({
                        "speaker_name": current_speaker,
                        "speaker_role": current_role,
                        "time": current_timestamp
                    })
                    
                    # drop the speaker line itself
                    text = re.sub(r'^.*?:\s*', '', text, count=1).strip()
                
                # Add to current speaker buffer
                speaker_buf.append(text)
                meta_buf = metadata
            
            # Tables should be kept whole
            elif element_type == "Table":
                doc = Document(
                    page_content=text,
                    metadata={**metadata, "is_table": True}
                )
                processed_docs.append(doc)
        
        # Add any remaining speaker buffer
        if speaker_buf and meta_buf:
            doc = Document(
                page_content="\n".join(speaker_buf),
                metadata=meta_buf
            )
            processed_docs.append(doc)
    
    return processed_docs 