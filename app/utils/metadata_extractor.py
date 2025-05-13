import re
import os
from typing import Dict, Any
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader

def extract_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from the earnings call PDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        "title": os.path.basename(file_path),
        "company": "",
        "date": "",
        "quarter": ""
    }
    
    # Load the first few pages of the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    
    if not pages:
        return metadata
    
    # Extract text from first page
    text = pages[0].page_content if pages else ""
    
    # Extract company name (usually in the header)
    company_matches = re.findall(r"([A-Z][A-Za-z\s]+)\s+(?:Inc\.|Ltd\.|Limited|Corp\.|Corporation)", text)
    if company_matches:
        metadata["company"] = company_matches[0].strip()
    
    # Extract date
    date_patterns = [
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
        r"\d{1,2}/\d{1,2}/\d{4}",
        r"\d{4}-\d{1,2}-\d{1,2}"
    ]
    
    for pattern in date_patterns:
        date_matches = re.findall(pattern, text)
        if date_matches:
            metadata["date"] = date_matches[0]
            break
    
    # Extract quarter information
    quarter_patterns = [
        r"Q[1-4]\s+\d{4}",
        r"(?:First|Second|Third|Fourth) Quarter\s+\d{4}",
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}"
    ]
    
    for pattern in quarter_patterns:
        quarter_matches = re.findall(pattern, text)
        if quarter_matches:
            metadata["quarter"] = quarter_matches[0]
            break
    
    # If we couldn't find the quarter but have a date, infer it
    if not metadata["quarter"] and metadata["date"]:
        try:
            # Try to parse the date
            for date_format in ["%B %d, %Y", "%d %B %Y", "%m/%d/%Y", "%Y-%m-%d"]:
                try:
                    date_obj = datetime.strptime(metadata["date"], date_format)
                    month = date_obj.month
                    quarter = ((month - 1) // 3) + 1
                    metadata["quarter"] = f"Q{quarter} {date_obj.year}"
                    break
                except ValueError:
                    continue
        except Exception:
            # If any error occurs during parsing, just skip
            pass
    
    return metadata 