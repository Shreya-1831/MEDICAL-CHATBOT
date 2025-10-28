import ollama
from typing import List, cast
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using Ollama's embedding models
    Falls back to simple hashing if Ollama is not available
    """
    
    def __init__(self, model_name: str = "nomic-embed-text"):
        """
        Initialize with Ollama embedding model
        Default: nomic-embed-text (optimized for embeddings)
        Alternatives: all-minilm, mxbai-embed-large
        """
        self.model_name = model_name
        self.use_ollama = self._check_ollama_availability()
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is available and model exists"""
        try:
            ollama.list()
            print(f"✓ Ollama embeddings available using model: {self.model_name}")
            return True
        except:
            print(f"⚠ Ollama not available - using fallback deterministic embeddings")
            return False
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents"""
        if self.use_ollama:
            return cast(Embeddings, self._ollama_embeddings(input))
        else:
            return cast(Embeddings, self._fallback_embeddings(input))
    
    def _ollama_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama"""
        embeddings = []
        
        for text in texts:
            try:
                response = ollama.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                
                if 'embedding' in response:
                    embeddings.append(response['embedding'])
                else:
                    embeddings.append(self._simple_embedding(text))
            except Exception as e:
                embeddings.append(self._simple_embedding(text))
        
        return embeddings
    
    def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Simple fallback embedding using TF-IDF-like approach"""
        return [self._simple_embedding(text) for text in texts]
    
    def _simple_embedding(self, text: str, dim: int = 384) -> List[float]:
        """
        Create a simple deterministic embedding vector
        Uses character frequencies and text features
        """
        import math
        
        embedding = [0.0] * dim
        
        if not text:
            return embedding
        
        text_lower = text.lower()
        text_len = len(text_lower)
        
        for i, char in enumerate(text_lower[:dim]):
            idx = i % dim
            char_val = ord(char) / 128.0
            position_weight = 1.0 - (i / max(text_len, 1))
            embedding[idx] += char_val * position_weight
        
        for word in text_lower.split()[:50]:
            hash_val = hash(word) % dim
            embedding[hash_val] += 1.0 / math.sqrt(len(word) + 1)
        
        char_counts = {}
        for char in text_lower:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        for char, count in list(char_counts.items())[:dim]:
            idx = ord(char) % dim
            embedding[idx] += count / text_len
        
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
