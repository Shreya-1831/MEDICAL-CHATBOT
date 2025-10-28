# RAG ML System with Ollama LLM

A comprehensive Retrieval Augmented Generation (RAG) system that processes multiple document types (PDFs, text files, images) and uses Ollama LLM for intelligent question answering.

## Features

### Multi-Modal Document Processing
- **PDF Support**: Extract text from PDF files (books, thesis, research papers)
- **Text Files**: Process plain text documents
- **Image OCR**: Extract text from images using Tesseract OCR
- **Smart Chunking**: Intelligent text chunking with overlap for better context

### Vector Database & Embeddings
- **ChromaDB**: Efficient vector database with cosine similarity search
- **Custom Embeddings**: Uses Ollama embeddings (nomic-embed-text) with fallback to deterministic embeddings
- **Persistent Storage**: Vector database persists across sessions
- **No External Dependencies**: Works without sentence-transformers installation

### RAG Pipeline
- **Retrieval**: Semantic search to find relevant document chunks
- **Context Building**: Assembles context from top-k similar chunks
- **Generation**: Uses Ollama LLM for answer generation
- **Multiple Models**: Support for llama2, mistral, codellama, llama3, phi

### User Interface
- **Document Upload**: Drag-and-drop interface for multiple files
- **Query Interface**: Text input and image upload for questions
- **Context Display**: View retrieved chunks used for answer generation
- **System Stats**: Monitor uploaded documents and system status

## System Architecture

```
User Input (Text/Image)
        ↓
Document Processor
  - PDF Extraction (PyPDF2)
  - OCR (Pytesseract)
  - Text Chunking
        ↓
Vector Embeddings
  - Ollama (nomic-embed-text)
  - Fallback (TF-IDF-like)
        ↓
ChromaDB (Vector Storage)
        ↓
Query Processing
  - Similarity Search
  - Top-K Retrieval
        ↓
Ollama LLM
  - Context + Query
  - Answer Generation
        ↓
User Response
```

## Components

### 1. Document Processor (`document_processor.py`)
- Handles PDF, TXT, and image files
- Extracts text using PyPDF2 and Tesseract OCR
- Implements chunking strategies with configurable size and overlap
- Supports recursive character splitting for better semantic chunks

### 2. Vector Database (`vector_db.py`)
- ChromaDB integration with persistent storage
- Cosine similarity for semantic search
- Metadata tracking (source, chunk ID)
- Collection management and statistics

### 3. RAG Pipeline (`rag_pipeline.py`)
- Query processing and context retrieval
- Prompt engineering for better answers
- Ollama LLM integration
- Support for streaming responses

### 4. Streamlit App (`app.py`)
- Document upload and management
- Query interface with text and image support
- Retrieved context visualization
- System information dashboard

## How to Use

### Prerequisites
Make sure Ollama is installed and running locally with at least one model:

```bash
# Install Ollama (if not already installed)
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2
# or
ollama pull mistral
```

### Using the System

1. **Upload Documents**
   - Click "Upload Documents" in the sidebar
   - Select PDFs, text files, or images
   - Click "Process Documents" to extract and vectorize text

2. **Ask Questions**
   - Go to "Query Interface" tab
   - Enter your question in the text area
   - Optionally upload an image to extract text
   - Click "Search" to get answers

3. **View Context**
   - Expand "View Retrieved Context" to see which document chunks were used
   - Check similarity scores to understand relevance

4. **Manage Documents**
   - View uploaded documents in the sidebar
   - See chunk counts and document types
   - Clear all documents to start fresh

## Configuration

### Model Settings
- **Ollama Model**: Select from available models (llama2, mistral, etc.)
- **Top K Results**: Number of chunks to retrieve (1-10)
- **Chunk Size**: Default 500 characters (configurable in code)
- **Chunk Overlap**: Default 50 characters (configurable in code)

## Technical Details

### ML/DL Models Used

1. **Embedding Model**: Custom Ollama Embeddings
   - Primary: Ollama's nomic-embed-text model (optimized for semantic search)
   - Fallback: Deterministic TF-IDF-like embeddings (when Ollama unavailable)
   - Generates 384-dimensional embeddings
   - No external API dependencies

2. **OCR Model**: Tesseract
   - Traditional computer vision approach
   - Extracts text from images and scanned documents

3. **LLM**: Ollama (Local)
   - Supports multiple models (llama2, mistral, codellama, etc.)
   - Local inference (no API keys needed)
   - Customizable temperature and parameters

### RAG Workflow

1. **Document Ingestion**
   - Extract text from various formats
   - Split into overlapping chunks
   - Generate embeddings
   - Store in vector database

2. **Query Processing**
   - Convert query to embedding
   - Perform similarity search
   - Retrieve top-k relevant chunks
   - Build context from retrieved chunks

3. **Answer Generation**
   - Create prompt with context + query
   - Send to Ollama LLM
   - Generate contextual answer
   - Return answer with sources

### Data Flow

```
Input Documents → Text Extraction → Chunking → Embeddings → ChromaDB
                                                                ↓
User Query → Embedding → Similarity Search → Top-K Chunks → Context
                                                                ↓
Context + Query → Prompt → Ollama LLM → Answer → User
```

## Advantages of This Approach

1. **Multi-Modal Support**: Handle text, PDFs, and images in one system
2. **Local Processing**: No external API dependencies, runs entirely locally
3. **Persistent Storage**: Vector database persists, no need to re-process documents
4. **Scalable**: Can handle large document collections
5. **Transparent**: Shows retrieved context and similarity scores
6. **Customizable**: Easy to swap models, adjust parameters, modify prompts

## Future Enhancements

- Add support for more document types (DOCX, HTML, Markdown)
- Implement semantic chunking based on document structure
- Add metadata filtering (by date, document type, author)
- Support for conversation history and follow-up questions
- Batch processing for large document collections
- Multi-modal embeddings using CLIP for image-text retrieval
- Fine-tuning embeddings on domain-specific data

## Troubleshooting

### Common Issues

1. **"Ollama error"**: Make sure Ollama is running (`ollama serve`)
2. **"Model not found"**: Install the model with `ollama pull <model-name>`
3. **OCR not working**: Verify Tesseract is installed properly
4. **No documents found**: Upload and process documents first

## License

This is an educational project demonstrating RAG implementation with Ollama and ChromaDB.
