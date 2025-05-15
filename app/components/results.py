import streamlit as st
from app.utils.cleanup_utils import clean_document_data

def render_results(results):
    """
    Render query results with citations in a clean, simple UI.
    
    Args:
        results: Dictionary containing answer and citation information
    """
    if not results:
        return
    
    # Set page styles matching the screenshot
    st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    
    .quote-container {
        background-color: #1a1a1a;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        margin: 15px 0;
        border-radius: 4px;
        line-height: 1.6;
    }
    
    .source-info {
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }
    
    .source-label {
        font-weight: 600;
        min-width: 80px;
        color: #888;
    }
    
    .source-value {
        color: #fff;
    }
    
    .clear-button {
        margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display the answer
    st.markdown("## Answer")
    st.write(results["answer"])
    
    # Display citation if available
    if results.get("citation"):
        citation = results["citation"]
        
        # Display the original quote
        st.markdown("## Original Quote")
        citation_text = citation.get("text", "").strip()
        if citation_text:
            st.markdown(f"<div class='quote-container'>{citation_text}</div>", unsafe_allow_html=True)
        
        # Source section with simple layout
        st.markdown("## Source")
        
        # Create clean source info layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Speaker information
            speaker_name = citation.get("speaker_name", "Narrator")
            speaker_role = citation.get("speaker_role", "")
            
            if speaker_role:
                speaker_display = f"{speaker_name} ({speaker_role})"
            else:
                speaker_display = speaker_name
                
            st.markdown("**Speaker:**")
            
            # Page number
            st.markdown("**Page:**")
            
            # Section information
            if citation.get("section"):
                st.markdown("**Section:**")
            
            # Time information (if available)
            if citation.get("time"):
                st.markdown("**Time:**")
        
        with col2:
            # Speaker value
            st.markdown(f"{speaker_display}")
            
            # Page value
            page_num = citation.get("page", "")
            st.markdown(f"{page_num}")
            
            # Section value
            if citation.get("section"):
                st.markdown(f"{citation['section']}")
            
            # Time value (if available)
            if citation.get("time"):
                st.markdown(f"{citation['time']}")
        
        # Document details in collapsible section
        with st.expander("Document Details", expanded=False):
            doc_info = results["document"]
            if doc_info.get("title"):
                st.write(f"**File:** {doc_info['title']}")
            if doc_info.get("company"):
                st.write(f"**Company:** {doc_info['company']}")
            if doc_info.get("date"):
                st.write(f"**Date:** {doc_info['date']}")
            if doc_info.get("quarter"):
                st.write(f"**Period:** {doc_info['quarter']}")
        
            # Add chunk ID info for debugging
            if citation.get("chunk_id"):
                st.write(f"**Chunk ID:** {citation['chunk_id']}")
        
        # Clear document button at the bottom (cleanup metadata and vectorstore)
        if st.button("Clear document", key="clear_doc_button"):
            if 'doc_id' in st.session_state:
                doc_id = st.session_state['doc_id']
                # Use the new cleanup utility
                success = clean_document_data(doc_id)
                
                if success:
                    st.success("Document cleared successfully")
                else:
                    st.warning("Some document data may not have been fully cleared")
                
                # Clear session state and rerun
                del st.session_state.doc_id
                st.rerun()