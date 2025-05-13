import os
from typing import Union
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.config.settings import UPLOAD_DIRECTORY

def save_uploaded_file(uploaded_file: UploadedFile) -> str:
    """
    Save an uploaded file to disk and return its path.
    
    Args:
        uploaded_file: Streamlit uploaded file
        
    Returns:
        Path to the saved file
    """
    # Ensure upload directory exists
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
    
    # Create file path
    file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    return file_path 