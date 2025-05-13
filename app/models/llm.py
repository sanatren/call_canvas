import os
from typing import Any, Optional, Dict

from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub

from app.config.settings import DEFAULT_LLM_TYPE, OPENAI_API_KEY, HUGGINGFACE_API_KEY

def get_llm(llm_type: Optional[str] = None) -> LLM:
    """
    Get a configured LLM.
    
    Args:
        llm_type: Type of LLM to use (optional, defaults to config setting)
        
    Returns:
        Configured LLM
    """
    # If llm_type is None or invalid, use the default
    if not llm_type or llm_type not in ["openai", "huggingface"]:
        llm_type = DEFAULT_LLM_TYPE
    
    if llm_type == "openai":
        return _get_openai_llm()
    elif llm_type == "huggingface":
        return _get_huggingface_llm()
    else:
        # This shouldn't happen due to the check above, but just in case
        return _get_openai_llm()  # Fallback to OpenAI

def _get_openai_llm() -> LLM:
    """Get OpenAI LLM."""
    # Check for API key
    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found")
    
    # Configure the LLM with parameters optimized for our use case
    return ChatOpenAI(
        model_name="gpt-4", 
        temperature=0.1,  # Low temperature for factual responses
        api_key=api_key,
        model_kwargs={
            "response_format": {"type": "text"}
        }
    )

def _get_huggingface_llm() -> LLM:
    """Get HuggingFace LLM."""
    # Check for API key
    api_key = HUGGINGFACE_API_KEY or os.environ.get("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HuggingFace API key not found")
    
    # Configure the LLM with model suitable for our use case
    return HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # Example model
        huggingfacehub_api_token=api_key,
        model_kwargs={
            "temperature": 0.1,
            "max_length": 512,
            "max_new_tokens": 512
        }
    ) 