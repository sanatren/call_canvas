# Set protobuf implementation to pure Python to avoid descriptor issues
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import os
import sys

# Add the current directory to path to find the app module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.document_processor import DocumentProcessor
from app.core.query_engine import QueryEngine
from app.components.sidebar import render_sidebar
from app.components.results import render_results
from app.utils.file_utils import save_uploaded_file
from app.utils.cleanup_utils import clean_document_data

# Ensure static directory exists
static_dir = os.path.join(os.path.dirname(__file__), "app", "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# App configuration
st.set_page_config(
    page_title="CallCanvas", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CallCanvas - AI-powered earnings call insights"
    }
)

# Initialize app state
def init_app_state():
    """Initialize app state and optionally clean up old data."""
    # Check if we should clean up on startup
    # Set this to False in production
    CLEAN_ON_STARTUP = False
    
    # First-time initialization flag
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
        # Clear any stale data on startup
        if CLEAN_ON_STARTUP:
            clean_document_data()
            # Don't show a message as it would appear every time

# Load CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "app", "static", "style.css")
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    # Initialize app state
    init_app_state()
    
    # Load CSS
    load_css()
    
    # Hide Streamlit branding
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: flex !important;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("CallCanvas")
        st.markdown("AI-powered earnings call insights")
        render_sidebar()
    
    # Main area
    if 'doc_id' not in st.session_state:
        # File upload section when no document is loaded
        st.title("CallCanvas")
        st.subheader("Upload a transcript (PDF, max 10MB)")
        uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'], label_visibility="collapsed")
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                # Save the file
                file_path = save_uploaded_file(uploaded_file)
                
                # Process the document
                try:
                    doc_processor = DocumentProcessor()
                    doc_id = doc_processor.process(file_path)
                    
                    st.session_state.doc_id = doc_id
                    st.success("Document processed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    else:
        # Question and answer section when document is loaded
        st.markdown("## Ask a question about the transcript")
        
        # Query input
        query = st.text_input(
            "Question", 
            placeholder="e.g., Who built Jio's 4G and 5G network?",
            key="question_input",
            label_visibility="collapsed"
        )
        
        if query:
            with st.spinner("Finding answer..."):
                try:
                    query_engine = QueryEngine()
                    results = query_engine.query(st.session_state.doc_id, query)
                    
                    render_results(results)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
        
        # Add button to clear current document
        if st.button("Clear document", key="clear_doc"):
            if 'doc_id' in st.session_state:
                doc_id = st.session_state['doc_id']
                # Use the new cleanup utility
                success = clean_document_data(doc_id)
                
                if success:
                    st.success("Document cleared successfully!")
                else:
                    st.warning("Some document data may not have been fully cleared")
                    
                # Clear session state and rerun
                del st.session_state.doc_id
                st.rerun()

if __name__ == "__main__":
    main() 