import streamlit as st
import os
from rag_pipeline import RAGPipeline
from pathlib import Path

# Configure Streamlit
st.set_page_config(
    page_title="RAG Pipeline with Claude Sonnet 4",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    """Load the RAG pipeline (cached for performance)."""
    return RAGPipeline()

def main():
    st.title("🤖 RAG Pipeline with Claude Sonnet 4")
    st.markdown("Ask questions about your documents!")
    
    # Initialize pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        st.info("Make sure your .env file is configured correctly with ANTHROPIC_API_KEY")
        return
    
    # Sidebar for configuration and file upload
    with st.sidebar:
        st.header("Configuration")
        k = st.slider("Number of documents to retrieve", 1, 10, 5)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
        
        st.header("Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files"):
                # Create temporary directory for uploaded files
                upload_dir = Path("./temp_uploads")
                upload_dir.mkdir(exist_ok=True)
                
                # Save uploaded files
                saved_files = []
                for uploaded_file in uploaded_files:
                    file_path = upload_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_files.append(file_path)
                
                # Process files
                with st.spinner("Processing documents..."):
                    try:
                        pipeline.ingest_documents(str(upload_dir))
                        st.success(f"Successfully processed {len(saved_files)} files!")
                        
                        # Clean up temporary files
                        for file_path in saved_files:
                            file_path.unlink()
                        upload_dir.rmdir()
                        
                    except Exception as e:
                        st.error(f"Error processing files: {e}")
        
        st.markdown("---")
        
        # Clear button
        if st.button("Clear All Documents", type="secondary"):
            pipeline.vector_store.clear_collection()
            st.success("All documents cleared!")
            st.rerun()
        
        st.header("Pipeline Stats")
        if st.button("Refresh Stats"):
            stats = pipeline.get_stats()
            
            st.metric("Documents", stats['vector_store']['count'])
            st.metric("Chunk Size", stats['chunk_size'])
            st.metric("Model", stats['model'])
            
            with st.expander("Detailed Stats"):
                st.json(stats)
    
    # Main interface
    st.header("Ask a Question")
    
    # Get collection info
    collection_info = pipeline.get_stats()['vector_store']
    
    if collection_info['count'] == 0:
        st.warning("No documents have been ingested yet. Please upload and process some documents first.")
        return
    
    st.info(f"Ready to answer questions based on {collection_info['count']} document chunks.")
    
    question = st.text_input(
        "Enter your question:", 
        placeholder="What would you like to know about your documents?",
        key="question_input"
    )
    
    if st.button("Ask Question", type="primary") and question:
        with st.spinner("Searching and generating answer..."):
            result = pipeline.query(question, k=k, temperature=temperature)
        
        # Display answer
        st.subheader("📝 Answer")
        st.write(result['answer'])
        
        # Display sources
        if result['sources']:
            st.subheader("📚 Sources")
            for source in result['sources']:
                st.write(f"• {source}")
        
        # Display retrieved documents (expandable)
        with st.expander(f"🔍 Retrieved Documents ({len(result['retrieved_docs'])} chunks)", expanded=False):
            for i, doc in enumerate(result['retrieved_docs'], 1):
                st.markdown(f"**Chunk {i}** from `{doc['metadata']['filename']}`:")
                
                # Show similarity score if available
                if doc.get('distance') is not None:
                    similarity = 1 - doc['distance']  # Convert distance to similarity
                    st.caption(f"Similarity: {similarity:.3f}")
                
                # Show content (truncated)
                content = doc['content']
                if len(content) > 500:
                    st.write(content[:500] + "...")
                    with st.expander("Show full chunk"):
                        st.write(content)
                else:
                    st.write(content)
                
                if i < len(result['retrieved_docs']):
                    st.markdown("---")
    
    # Example questions
    st.header("💡 Example Questions")
    example_questions = [
        "What are the main topics discussed in the documents?",
        "Can you summarize the key findings?",
        "What are the most important recommendations?",
        "What data or statistics are mentioned?",
        "Are there any conclusions or takeaways?"
    ]
    
    cols = st.columns(2)
    for i, eq in enumerate(example_questions):
        col = cols[i % 2]
        if col.button(eq, key=f"example_{i}"):
            st.rerun()

if __name__ == "__main__":
    main()