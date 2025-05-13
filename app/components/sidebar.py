import streamlit as st

from app.models.document import Document
from app.config.settings import DEFAULT_LLM_TYPE, EMBEDDING_MODEL

def render_sidebar():
    """Render the application sidebar."""
    # Title and description
    st.title("CallCanvas")
    st.markdown("AI-powered earnings call insights")
    
    # Description
    st.markdown("""
    CallCanvas helps you extract insights from earnings call transcripts. 
    Upload a transcript PDF and ask questions in natural language.
    """)
    
    # Settings section
    st.subheader("Settings")
    
    # LLM Selection
    if 'llm_type' not in st.session_state:
        st.session_state.llm_type = DEFAULT_LLM_TYPE
        
    llm_type = st.selectbox(
        "LLM Provider",
        options=["openai", "huggingface"],
        index=0 if st.session_state.llm_type == "openai" else 1,
        key="llm_selection"
    )
    
    # Save selected LLM type to session state
    st.session_state.llm_type = llm_type
    
    # Advanced settings (collapsible)
    with st.expander("Advanced Settings"):
        st.write(f"Embedding Model: {EMBEDDING_MODEL}")