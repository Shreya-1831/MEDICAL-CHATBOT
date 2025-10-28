import chromadb
from chromadb.config import Settings
from typing import List, Dict
from embedding_function import OllamaEmbeddingFunction

class VectorDatabase:
    def __init__(self, collection_name="documents", persist_directory="./chroma_db"):
        """Initialize ChromaDB vector database"""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize custom embedding function
        self.embedding_function = OllamaEmbeddingFunction()
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection with custom embedding function
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )
        
        self.doc_count = 0
    
    def add_documents(self, chunks: List[Dict[str, str]], source: str):
        """Add document chunks to the vector database"""
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            doc_id = f"{source}_{self.doc_count}_{chunk['chunk_id']}"
            documents.append(chunk['text'])
            metadatas.append({
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id']
            })
            ids.append(doc_id)
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        self.doc_count += 1
    
    def query(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """Query the vector database for similar documents"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        # Format results
        retrieved_chunks = []
        
        if results and results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                chunk = {
                    'text': results['documents'][0][i],
                    'source': results['metadatas'][0][i]['source'],
                    'chunk_id': results['metadatas'][0][i]['chunk_id'],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def clear_database(self):
        """Clear all documents from the database"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )
            self.doc_count = 0
        except Exception as e:
            print(f"Error clearing database: {str(e)}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name
            }
        except Exception as e:
            return {
                'error': str(e)
            }
