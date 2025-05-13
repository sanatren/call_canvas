from typing import Dict, Any, List
import re

import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from app.models.document import Document as DocModel
from app.embeddings.embeddings_manager import EmbeddingsManager
from app.models.llm import get_llm
from app.retrieval.retriever import get_retriever

class QueryEngine:
    """Handles user queries and retrieves answers from the document."""
    
    def __init__(self):
        self.embeddings_manager = EmbeddingsManager()
        # Use LLM type from session state if available, otherwise use default
        llm_type = st.session_state.get('llm_type')
        self.llm = get_llm(llm_type)
    
    def query(self, doc_id: str, query: str) -> Dict[str, Any]:
        """
        Process a user query and return results.
        
        Args:
            doc_id: ID of the document to query
            query: User question
            
        Returns:
            Dictionary with answer and citation information
        """
        # Get document metadata
        document = DocModel.get(doc_id)
        
        # Get retriever for this document
        retriever = get_retriever(doc_id, self.embeddings_manager)
        
        # Create QA chain with citations
        qa_chain = self._create_qa_chain_with_citations(retriever)
        
        # Get results
        raw_results = qa_chain({"query": query})
        
        # Extract the answer and source documents
        answer = raw_results.get("result", "")
        source_docs = raw_results.get("source_documents", [])
        
        # Attempt to extract page/line citation from the answer if present
        extracted_citation = self._extract_citation_from_text(answer)
        if extracted_citation:
            # If citation was found in answer text, remove it from the answer
            answer = re.sub(r'\(Page\s+\d+(?:,\s*Lines?\s*\d+(?:-\d+)?)?\)', '', answer).strip()
        
        # Format the results with proper citations
        results = self._format_results(answer, source_docs, document.metadata, extracted_citation)
        
        return results
    
    def _create_qa_chain_with_citations(self, retriever):
        """Create a QA chain that includes citations in the output."""
        # Create a custom prompt that asks for citations
        template = """You are an AI assistant providing accurate information about earnings calls.
        Use only the context provided to answer the question. If you don't know the answer, say "I don't have enough information to answer this question."
        
        Keep your answer concise and factual, focusing on numbers and exact figures.
        Provide a short, direct answer, followed by a citation to the page and line number where you found the information.
        
        Include the exact page and line numbers at the end of your answer like this: (Page X, Line Y)
        
        Context: {context}
        
        Question: {question}
        
        Answer (concise with page/line citation):"""
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        
        return qa_chain
    
    def _extract_citation_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract page and line citation from text if present.
        
        Args:
            text: Answer text potentially containing citation
            
        Returns:
            Dictionary with page and line information if found, None otherwise
        """
        # Look for citation patterns like (Page 7) or (Page 7, Lines 2-3)
        citation_match = re.search(r'\(Page\s+(\d+)(?:,\s*Lines?\s+(\d+)(?:-\d+)?)?\)', text)
        if citation_match:
            page = int(citation_match.group(1))
            line = int(citation_match.group(2)) if citation_match.group(2) else 1
            return {"page": page, "line_number": line}
        return None
    
    def _format_results(self, answer: str, source_docs: List[Document], 
                       doc_metadata: Dict[str, Any], extracted_citation: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format the results with proper citations.
        
        Args:
            answer: Generated answer
            source_docs: Source documents used for the answer
            doc_metadata: Metadata about the document
            extracted_citation: Citation extracted from answer text (if any)
            
        Returns:
            Formatted results with citations
        """
        # Extract the best citation from source documents
        citation = None
        if source_docs:
            # Find the best source document with most relevant information
            best_source = self._find_best_source_document(source_docs)
            
            # Extract the text and metadata
            quote = best_source.page_content.strip()
            metadata = best_source.metadata
            
            # Use extracted citation if available (from answer text), otherwise use metadata from retrieved documents
            if extracted_citation:
                page = extracted_citation.get("page")
                line_number = extracted_citation.get("line_number")
                
                # Validate page number against available metadata
                if page > metadata.get("total_pages", 1000):  # Use a high number as fallback
                    page = metadata.get("page", 1)
            else:
                # Use metadata from the best matching document
                page = metadata.get("page", 1)
                line_number = metadata.get("line_number", 1)
            
            # Get speaker information
            speaker_name = metadata.get("speaker_name", "Narrator")
            speaker_role = metadata.get("speaker_role", "")
            time_stamp = metadata.get("time", "")
            
            # Attempt to identify section
            section = self._identify_section(best_source, source_docs)
            
            citation = {
                "text": quote,
                "page": page,
                "line_number": line_number,
                "speaker_name": speaker_name,
                "speaker_role": speaker_role,
                "time": time_stamp,
                "section": section
            }
        
        # Format the final result
        formatted_result = {
            "answer": answer,
            "citation": citation,
            "document": {
                "title": doc_metadata.get("title", ""),
                "company": doc_metadata.get("company", ""),
                "date": doc_metadata.get("date", ""),
                "quarter": doc_metadata.get("quarter", "")
            }
        }
        
        return formatted_result
    
    def _identify_section(self, main_doc: Document, all_docs: List[Document]) -> str:
        """
        Attempt to identify the section of the document.
        
        Args:
            main_doc: The document to identify section for
            all_docs: All retrieved documents for context
            
        Returns:
            Section name if identifiable, empty string otherwise
        """
        # Common section patterns in earnings calls
        section_patterns = {
            r"(?i)opening remarks|introduction|welcome": "Opening Remarks",
            r"(?i)financial(?:\s+results|\s+performance|\s+highlights)": "Financial Results",
            r"(?i)q(?:uestion)?(?:\s+|&|\s*and\s*)a(?:nswer)?": "Q&A",
            r"(?i)guidance|outlook|forecast": "Guidance & Outlook",
            r"(?i)operational|operations|business\s+update": "Operational Update",
            r"(?i)closing\s+remarks": "Closing Remarks"
        }
        
        # First check in main doc
        text = main_doc.page_content
        for pattern, section_name in section_patterns.items():
            if re.search(pattern, text):
                return section_name
        
        # Then check in nearby docs for context
        nearby_text = " ".join([doc.page_content for doc in all_docs[:3]])
        for pattern, section_name in section_patterns.items():
            if re.search(pattern, nearby_text):
                return section_name
        
        return ""
    
    def _find_best_source_document(self, source_docs: List[Document]) -> Document:
        """
        Find the best source document with the most complete metadata.
        Prioritize documents with specific information.
        
        Args:
            source_docs: List of source documents
            
        Returns:
            The best source document
        """
        if not source_docs:
            return None
        
        # Score each document based on metadata completeness
        scored_docs = []
        for doc in source_docs:
            score = 0
            metadata = doc.metadata
            
            # Prioritize docs with specific information
            if metadata.get("time"):
                score += 5
            if metadata.get("speaker_name") and metadata.get("speaker_name") != "Narrator":
                score += 4
            if metadata.get("speaker_role"):
                score += 3
            if metadata.get("page") and metadata.get("line_number"):
                score += 2
            if not metadata.get("is_full_page", False):  # Prefer specific lines over full pages
                score += 1
                
            scored_docs.append((score, doc))
        
        # Sort by score (highest first) using only the score to avoid comparing Document objects
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        
        # Return the highest scoring document
        return scored_docs[0][1] if scored_docs else source_docs[0]