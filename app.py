import streamlit as st
import os
from document_processor import DocumentProcessor
from vector_db import VectorDatabase
from rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="RAG ML System with Ollama",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = VectorDatabase()
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_db)
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []

# Title and description
st.title("🤖 RAG ML System with Ollama LLM")
st.markdown("""
This system uses **Retrieval Augmented Generation (RAG)** to answer questions based on your uploaded documents.
Upload PDFs, text files, or images (with OCR) to build your knowledge base.
""")

# Sidebar for document upload and management
with st.sidebar:
    st.header("📚 Document Management")
    
    # Document upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'txt', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload PDFs, text files, or images for OCR processing"
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0)
                total_files = len(uploaded_files)
                
                for idx, file in enumerate(uploaded_files):
                    # Save uploaded file temporarily
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Process document
                    try:
                        chunks = st.session_state.doc_processor.process_document(
                            temp_path, 
                            file.name
                        )
                        
                        # Add to vector database
                        st.session_state.vector_db.add_documents(chunks, file.name)
                        
                        st.session_state.uploaded_docs.append({
                            'name': file.name,
                            'chunks': len(chunks),
                            'type': file.type
                        })
                        
                        st.success(f"✓ {file.name}: {len(chunks)} chunks processed")
                    except Exception as e:
                        st.error(f"✗ Error processing {file.name}: {str(e)}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    
                    progress_bar.progress((idx + 1) / total_files)
                
                st.success(f"Processed {total_files} documents!")
    
    # Display uploaded documents
    if st.session_state.uploaded_docs:
        st.subheader("Uploaded Documents")
        for doc in st.session_state.uploaded_docs:
            st.text(f"📄 {doc['name']}")
            st.caption(f"   {doc['chunks']} chunks")
        
        if st.button("Clear All Documents"):
            st.session_state.vector_db.clear_database()
            st.session_state.uploaded_docs = []
            st.rerun()
    
    # Model settings
    st.header("⚙️ Model Settings")
    ollama_model = st.selectbox(
        "Ollama Model",
        ["llama2", "mistral", "codellama", "llama3", "phi"],
        help="Select the Ollama model to use for generation"
    )
    
    top_k = st.slider(
        "Top K Results",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of relevant chunks to retrieve"
    )
    
    st.session_state.rag_pipeline.set_model(ollama_model)
    st.session_state.rag_pipeline.set_top_k(top_k)

# Main content area
tab1, tab2 = st.tabs(["💬 Query Interface", "📊 System Info"])

with tab1:
    st.header("Ask Questions About Your Documents")
    
    # Query input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="What would you like to know about the uploaded documents?"
        )
    
    with col2:
        st.write("")
        st.write("")
        query_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Image upload for query
    query_image = st.file_uploader(
        "Or upload an image to extract text and search:",
        type=['png', 'jpg', 'jpeg'],
        key="query_image"
    )
    
    if query_image:
        st.image(query_image, caption="Uploaded Image", width=300)
        if st.button("Extract Text from Image"):
            with st.spinner("Extracting text from image..."):
                temp_image_path = f"temp_query_{query_image.name}"
                with open(temp_image_path, "wb") as f:
                    f.write(query_image.getbuffer())
                
                extracted_text = st.session_state.doc_processor.extract_text_from_image(temp_image_path)
                st.text_area("Extracted Text:", value=extracted_text, height=150)
                
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
    
    # Process query
    if query_button and query:
        if not st.session_state.uploaded_docs:
            st.warning("⚠️ Please upload some documents first!")
        else:
            with st.spinner("Generating answer..."):
                try:
                    # Get response from RAG pipeline
                    response, retrieved_chunks = st.session_state.rag_pipeline.query(query)
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.markdown(f"**{response}**")
                    
                    # Display retrieved context
                    with st.expander("📖 View Retrieved Context"):
                        for i, chunk in enumerate(retrieved_chunks, 1):
                            distance = chunk.get('distance')
                            distance_str = f"{distance:.4f}" if distance is not None else "N/A"
                            st.markdown(f"**Chunk {i}** (Distance: {distance_str})")
                            st.text(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                            st.caption(f"Source: {chunk.get('source', 'Unknown')}")
                            st.divider()
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure Ollama is running locally with the selected model installed.")

with tab2:
    st.header("System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", len(st.session_state.uploaded_docs))
    
    with col2:
        total_chunks = sum(doc['chunks'] for doc in st.session_state.uploaded_docs)
        st.metric("Total Chunks", total_chunks)
    
    with col3:
        st.metric("Vector DB", "ChromaDB")
    
    st.subheader("📋 Document Details")
    if st.session_state.uploaded_docs:
        import pandas as pd
        df = pd.DataFrame(st.session_state.uploaded_docs)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No documents uploaded yet.")
    
    st.subheader("🔧 System Components")
    st.markdown("""
    - **Document Processing**: PyPDF2 (PDF), Pytesseract (OCR for images)
    - **Text Chunking**: Recursive character splitting
    - **Vector Embeddings**: Custom Ollama embeddings (nomic-embed-text) with fallback
    - **Vector Database**: ChromaDB with cosine similarity
    - **LLM**: Ollama (Local inference)
    - **RAG Framework**: Custom pipeline with LangChain integration
    """)
    
    st.subheader("💡 How It Works")
    st.markdown("""
    1. **Document Upload**: Upload PDFs, text files, or images
    2. **Text Extraction**: Extract text using PyPDF2 or OCR (Tesseract)
    3. **Chunking**: Split text into manageable chunks (500 characters)
    4. **Embedding**: Convert chunks to vector embeddings
    5. **Storage**: Store embeddings in ChromaDB vector database
    6. **Query**: User asks a question
    7. **Retrieval**: Find most similar chunks using cosine similarity
    8. **Generation**: Ollama LLM generates answer based on retrieved context
    """)

# Footer
st.markdown("---")
st.caption("RAG ML System powered by Ollama, ChromaDB, and Streamlit")
