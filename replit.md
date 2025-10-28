# RAG ML System with Ollama LLM

## Overview

This is a Retrieval Augmented Generation (RAG) system that processes multiple document types (PDFs, text files, images) and uses Ollama LLM for intelligent question answering. The system extracts text from various formats, converts them to vector embeddings, stores them in a persistent vector database, and retrieves relevant context to generate accurate answers to user queries.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with drag-and-drop file upload interface
- **Session Management**: Uses Streamlit session state to maintain vector database, RAG pipeline, document processor, and uploaded document list across user interactions
- **UI Components**: 
  - Main area for query input and results display
  - Sidebar for document management and upload
  - Progress indicators for document processing
  - Context display showing retrieved chunks used for answer generation

### Backend Architecture

**Document Processing Pipeline**
- **Multi-format Support**: Handles PDFs (PyPDF2), plain text files, and images (OCR via Pytesseract)
- **Chunking Strategy**: Intelligent text splitting with configurable chunk size (500 chars default) and overlap (50 chars default) to preserve context across boundaries
- **Page Tracking**: PDF processing maintains page number metadata for source attribution

**Embedding & Vector Storage**
- **Primary Embeddings**: Ollama's nomic-embed-text model for high-quality semantic embeddings
- **Fallback Mechanism**: Deterministic TF-IDF-like embeddings when Ollama is unavailable, ensuring system works without external dependencies
- **Vector Database**: ChromaDB with cosine similarity metric for semantic search
- **Persistence**: Vector database stored locally in `./chroma_db` directory, persists across sessions

**RAG Pipeline Flow**
1. User submits query (text or image via OCR)
2. Query is embedded using same embedding function as documents
3. Vector similarity search retrieves top-k most relevant chunks (default k=3)
4. Retrieved chunks assembled into context with source attribution
5. Prompt constructed combining user query and retrieved context
6. Ollama LLM generates answer based on grounded context
7. Response returned with source chunks for transparency

**LLM Integration**
- **Primary Interface**: Ollama API for local LLM inference
- **Model Flexibility**: Supports multiple Ollama models (llama2, mistral, codellama, llama3, phi)
- **Default Model**: llama2
- **Error Handling**: Graceful degradation with informative error messages when Ollama unavailable

### External Dependencies

**Third-party Libraries**
- `streamlit`: Web application framework for user interface
- `PyPDF2`: PDF text extraction
- `pytesseract`: OCR engine wrapper for image text extraction (requires Tesseract binary)
- `Pillow (PIL)`: Image processing for OCR pipeline
- `chromadb`: Vector database for semantic search
- `ollama`: Python client for Ollama LLM API

**External Services**
- **Ollama**: Local LLM inference server (must be running separately)
  - Used for: Text embeddings (nomic-embed-text model) and answer generation (configurable model)
  - Fallback: System continues with deterministic embeddings if unavailable
- **Tesseract OCR**: System-level OCR engine for image text extraction

**Database**
- **ChromaDB**: Embedded vector database with local file-based persistence
  - Storage Location: `./chroma_db` directory
  - Collection: "documents" (configurable)
  - Similarity Metric: Cosine similarity
  - No external database server required

**Key Architectural Decisions**

1. **Local-First Architecture**: All components run locally without cloud dependencies, ensuring privacy and offline capability
2. **Graceful Degradation**: Fallback embedding mechanism ensures core functionality works even when Ollama is unavailable
3. **Persistent Storage**: Vector database persists across sessions, avoiding need to reprocess documents
4. **Modular Design**: Separate classes for document processing, embeddings, vector database, and RAG pipeline enable easy testing and extension
5. **Stateful Session Management**: Streamlit session state maintains initialized components, improving performance by avoiding repeated initialization