import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
import os

class VectorStore:
    def __init__(self, 
                 db_path: str = "./chroma_db",
                 collection_name: str = "documents",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(collection_name)
            print(f"Created new collection: {collection_name}")
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the vector store."""
        print(f"Processing {len(chunks)} chunks...")
        
        # Check if we have any chunks
        if not chunks:
            print("No chunks to process")
            return
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk['content'])
            metadatas.append({
                'source': chunk['source'],
                'filename': chunk['filename'],
                'chunk_id': chunk['chunk_id'],
                'total_chunks': chunk['total_chunks']
            })
            ids.append(f"chunk_{i}_{chunk['filename']}_{chunk['chunk_id']}")
        
        # Generate embeddings in batches to avoid memory issues
        print("Generating embeddings...")
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_docs, show_progress_bar=True)
            all_embeddings.extend(batch_embeddings)
        
        # Add to ChromaDB
        print("Adding to vector store...")
        self.collection.add(
            embeddings=[embedding.tolist() for embedding in all_embeddings],
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Added {len(chunks)} chunks to vector store")
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )
        
        # Format results
        search_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                search_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'id': results['ids'][0][i]
                })
        
        return search_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            'name': self.collection_name,
            'count': count,
            'db_path': self.db_path
        }
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(self.collection_name)
            print(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            print(f"Error clearing collection: {e}")