"""
Easy Custom Model Integration
===============================

This file makes it simple to use your trained model with the RAG system.
Just update the configuration below and the system will automatically use your model.
"""

import os
from typing import List
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing import cast

class CustomModelEmbedding(EmbeddingFunction):
    """
    Drop-in replacement for embedding_function.py that uses YOUR custom model
    """
    
    def __init__(self):
        # ====== CONFIGURATION: Update these settings ======
        
        # Choose your model type: "sklearn", "huggingface", "pytorch", "ollama", "fallback"
        self.MODEL_TYPE = "sklearn"  # Change this to match your trained model
        
        # Path to your saved model
        self.MODEL_PATH = "./my_custom_model"  # Update with your model path
        
        # Embedding dimension (must match your model's output)
        self.EMBEDDING_DIM = 384
        
        # ================================================
        
        self.model = None
        self.is_loaded = False
        
        # Try to load your model
        self._load_model()
    
    def _load_model(self):
        """Load your custom model based on MODEL_TYPE"""
        
        print(f"\nAttempting to load {self.MODEL_TYPE} model from {self.MODEL_PATH}...")
        
        if self.MODEL_TYPE == "sklearn":
            self._load_sklearn_model()
        elif self.MODEL_TYPE == "huggingface":
            self._load_huggingface_model()
        elif self.MODEL_TYPE == "pytorch":
            self._load_pytorch_model()
        elif self.MODEL_TYPE == "ollama":
            self._load_ollama_model()
        elif self.MODEL_TYPE == "fallback":
            print("⚠ Using fallback embeddings (no model loaded)")
            self.is_loaded = False
        else:
            print(f"✗ Unknown model type: {self.MODEL_TYPE}")
            self.is_loaded = False
        
        if not self.is_loaded and self.MODEL_TYPE != "fallback":
            print("⚠ Falling back to deterministic embeddings")
    
    def _load_sklearn_model(self):
        """Load scikit-learn model (TF-IDF, etc.)"""
        try:
            import joblib
            
            if os.path.exists(self.MODEL_PATH):
                self.model = joblib.load(self.MODEL_PATH)
                self.is_loaded = True
                print(f"✓ Scikit-learn model loaded successfully")
            else:
                print(f"✗ Model file not found: {self.MODEL_PATH}")
                print("  Train a model using train_custom_model.py first")
                
        except Exception as e:
            print(f"✗ Error loading sklearn model: {str(e)}")
    
    def _load_huggingface_model(self):
        """Load HuggingFace model"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            if os.path.exists(self.MODEL_PATH):
                self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
                self.model = AutoModel.from_pretrained(self.MODEL_PATH)
                self.is_loaded = True
                print(f"✓ HuggingFace model loaded successfully")
            else:
                print(f"✗ Model directory not found: {self.MODEL_PATH}")
                
        except Exception as e:
            print(f"✗ Error loading HuggingFace model: {str(e)}")
    
    def _load_pytorch_model(self):
        """Load PyTorch model"""
        try:
            import torch
            
            if os.path.exists(self.MODEL_PATH):
                self.model = torch.load(self.MODEL_PATH, map_location='cpu')
                self.model.eval()
                self.is_loaded = True
                print(f"✓ PyTorch model loaded successfully")
            else:
                print(f"✗ Model file not found: {self.MODEL_PATH}")
                
        except Exception as e:
            print(f"✗ Error loading PyTorch model: {str(e)}")
    
    def _load_ollama_model(self):
        """Use Ollama embeddings"""
        try:
            import ollama
            ollama.list()
            self.model_name = "nomic-embed-text"
            self.is_loaded = True
            print(f"✓ Ollama embeddings available")
        except:
            print(f"✗ Ollama not available")
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for documents"""
        if self.is_loaded:
            if self.MODEL_TYPE == "sklearn":
                return cast(Embeddings, self._sklearn_embeddings(input))
            elif self.MODEL_TYPE == "huggingface":
                return cast(Embeddings, self._huggingface_embeddings(input))
            elif self.MODEL_TYPE == "pytorch":
                return cast(Embeddings, self._pytorch_embeddings(input))
            elif self.MODEL_TYPE == "ollama":
                return cast(Embeddings, self._ollama_embeddings(input))
        
        # Fallback
        return cast(Embeddings, self._fallback_embeddings(input))
    
    def _sklearn_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sklearn model"""
        try:
            embeddings = self.model.transform(texts).toarray().tolist()
            return embeddings
        except Exception as e:
            print(f"Error in sklearn embeddings: {str(e)}")
            return self._fallback_embeddings(texts)
    
    def _huggingface_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using HuggingFace model"""
        try:
            import torch
            
            embeddings = []
            for text in texts:
                with torch.no_grad():
                    inputs = self.tokenizer(
                        text,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    outputs = self.model(**inputs)
                    # Mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                    embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            print(f"Error in HuggingFace embeddings: {str(e)}")
            return self._fallback_embeddings(texts)
    
    def _pytorch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using PyTorch model"""
        try:
            import torch
            
            embeddings = []
            for text in texts:
                # Add your preprocessing here
                # This is a placeholder - replace with your model's forward pass
                with torch.no_grad():
                    # embedding = self.model(preprocessed_input)
                    pass
            
            return embeddings
        except Exception as e:
            print(f"Error in PyTorch embeddings: {str(e)}")
            return self._fallback_embeddings(texts)
    
    def _ollama_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama"""
        import ollama
        
        embeddings = []
        for text in texts:
            try:
                response = ollama.embeddings(model=self.model_name, prompt=text)
                if 'embedding' in response:
                    embeddings.append(response['embedding'])
                else:
                    embeddings.append(self._simple_embedding(text))
            except:
                embeddings.append(self._simple_embedding(text))
        
        return embeddings
    
    def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Simple fallback embeddings"""
        return [self._simple_embedding(text) for text in texts]
    
    def _simple_embedding(self, text: str) -> List[float]:
        """Deterministic fallback embedding"""
        import math
        
        embedding = [0.0] * self.EMBEDDING_DIM
        
        if not text:
            return embedding
        
        text_lower = text.lower()
        text_len = len(text_lower)
        
        # Character-based features
        for i, char in enumerate(text_lower[:self.EMBEDDING_DIM]):
            idx = i % self.EMBEDDING_DIM
            char_val = ord(char) / 128.0
            position_weight = 1.0 - (i / max(text_len, 1))
            embedding[idx] += char_val * position_weight
        
        # Word-based features
        for word in text_lower.split()[:50]:
            hash_val = hash(word) % self.EMBEDDING_DIM
            embedding[hash_val] += 1.0 / math.sqrt(len(word) + 1)
        
        # Normalize
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding


"""
QUICK START GUIDE:
==================

1. Train your model:
   python train_custom_model.py

2. Update the configuration above:
   - Set MODEL_TYPE to match your model ("sklearn", "huggingface", etc.)
   - Set MODEL_PATH to your saved model location
   
3. Replace the embedding function in vector_db.py:
   
   In vector_db.py, change:
       from embedding_function import OllamaEmbeddingFunction
   To:
       from use_custom_model import CustomModelEmbedding as OllamaEmbeddingFunction

4. Restart the Streamlit app

That's it! Your custom model is now integrated.
"""
