import ollama
from typing import List, Dict, Tuple
from vector_db import VectorDatabase

class RAGPipeline:
    def __init__(self, vector_db: VectorDatabase, model_name: str = "llama2"):
        """Initialize RAG pipeline with vector database and Ollama model"""
        self.vector_db = vector_db
        self.model_name = model_name
        self.top_k = 3
    
    def set_model(self, model_name: str):
        """Set the Ollama model to use"""
        self.model_name = model_name
    
    def set_top_k(self, top_k: int):
        """Set number of documents to retrieve"""
        self.top_k = top_k
    
    def query(self, user_query: str) -> Tuple[str, List[Dict]]:
        """
        Process a query through the RAG pipeline
        Returns: (answer, retrieved_chunks)
        """
        # Step 1: Retrieve relevant chunks from vector database
        retrieved_chunks = self.vector_db.query(user_query, top_k=self.top_k)
        
        if not retrieved_chunks:
            return "No relevant documents found. Please upload some documents first.", []
        
        # Step 2: Build context from retrieved chunks
        context = self._build_context(retrieved_chunks)
        
        # Step 3: Generate prompt with context
        prompt = self._build_prompt(user_query, context)
        
        # Step 4: Generate answer using Ollama
        try:
            answer = self._generate_answer(prompt)
        except Exception as e:
            answer = f"Error generating answer: {str(e)}. Make sure Ollama is running and the model '{self.model_name}' is installed."
        
        return answer, retrieved_chunks
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Document {i} - {chunk['source']}]\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for the LLM with context and query"""
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so
- Be concise and accurate
- Cite which document(s) you used to answer when relevant

Answer:"""
        
        return prompt
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using Ollama"""
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 500
                }
            )
            
            if 'response' in response:
                return response['response']
            else:
                return "Error: Unexpected response format from Ollama"
        
        except Exception as e:
            raise Exception(f"Ollama error: {str(e)}")
    
    def generate_with_streaming(self, user_query: str):
        """
        Generator function for streaming responses
        Yields: (chunk_text, retrieved_chunks) for the first chunk, then just chunk_text
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.vector_db.query(user_query, top_k=self.top_k)
        
        if not retrieved_chunks:
            yield "No relevant documents found. Please upload some documents first.", []
            return
        
        # Build context and prompt
        context = self._build_context(retrieved_chunks)
        prompt = self._build_prompt(user_query, context)
        
        # Stream response from Ollama
        first_chunk = True
        try:
            stream = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 500
                }
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    if first_chunk:
                        yield chunk['response'], retrieved_chunks
                        first_chunk = False
                    else:
                        yield chunk['response']
        
        except Exception as e:
            error_msg = f"Error: {str(e)}. Make sure Ollama is running and '{self.model_name}' is installed."
            if first_chunk:
                yield error_msg, []
            else:
                yield error_msg
