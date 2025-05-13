import streamlit as st
import os
from app.core.document_processor import DocumentProcessor
from app.core.query_engine import QueryEngine
from app.components.sidebar import render_sidebar
from app.components.results import render_results
from app.utils.file_utils import save_uploaded_file

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

# Load CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "app", "static", "style.css")
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
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
        uploaded_file = st.file_uploader("", type=['pdf'])
        
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
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    else:
        # Question and answer section when document is loaded
        st.markdown("## Ask a question about the transcript")
        
        # Query input
        query = st.text_input(
            "", 
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
                del st.session_state.doc_id
                st.rerun()

if __name__ == "__main__":
    main() 