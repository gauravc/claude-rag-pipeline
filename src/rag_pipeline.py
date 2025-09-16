import os
from typing import List, Dict, Any, Optional
from anthropic import Anthropic
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStore

# Load environment variables
load_dotenv()

class RAGPipeline:
    def __init__(self):
        # Initialize Anthropic client
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in your .env file.")
        
        self.client = Anthropic(api_key=api_key)
        
        # Initialize components
        self.doc_processor = DocumentProcessor(
            chunk_size=int(os.getenv('CHUNK_SIZE', 500)),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 50)),
            use_ocr=os.getenv('USE_OCR', 'true').lower() == 'true'
        )
        
        self.vector_store = VectorStore(
            db_path=os.getenv('CHROMA_DB_PATH', './chroma_db'),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        )
        
        self.model = "claude-sonnet-4-20250514"
    
    def ingest_documents(self, documents_path: str) -> None:
        """Ingest documents into the RAG pipeline."""
        print(f"Loading documents from {documents_path}...")
        
        # Load and process documents
        documents = self.doc_processor.load_documents(documents_path)
        if not documents:
            print("No documents found or processed successfully.")
            return
        
        print(f"Loaded {len(documents)} documents")
        
        # Chunk documents
        chunks = self.doc_processor.chunk_documents(documents)
        if not chunks:
            print("No chunks created from documents.")
            return
        
        print(f"Created {len(chunks)} chunks")
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        print("Document ingestion completed!")
    
    def query(self, 
              question: str, 
              k: int = 5, 
              temperature: float = 0.1) -> Dict[str, Any]:
        """Query the RAG pipeline."""
        
        # Check if we have any documents in the vector store
        collection_info = self.vector_store.get_collection_info()
        if collection_info['count'] == 0:
            return {
                'answer': "No documents have been ingested yet. Please run the ingest command first.",
                'sources': [],
                'retrieved_docs': []
            }
        
        # Retrieve relevant documents
        print(f"Searching for relevant documents...")
        relevant_docs = self.vector_store.similarity_search(question, k=k)
        
        if not relevant_docs:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'retrieved_docs': []
            }
        
        # Prepare context
        context = self._prepare_context(relevant_docs)
        
        # Generate prompt
        prompt = self._create_prompt(question, context)
        
        # Call Claude
        print("Generating response with Claude Sonnet 4...")
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.content[0].text
            
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': [],
                'retrieved_docs': relevant_docs
            }
        
        # Extract sources
        sources = list(set([doc['metadata']['filename'] for doc in relevant_docs]))
        
        return {
            'answer': answer,
            'sources': sources,
            'retrieved_docs': relevant_docs
        }
    
    def _prepare_context(self, docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"Source {i} ({doc['metadata']['filename']}):\n{doc['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create the prompt for Claude."""
        
        # Check if this seems like a utility bill analysis question
        is_bill_analysis = any(keyword in question.lower() for keyword in [
            'bill', 'energy', 'cost', 'amount', 'total', 'charge', 'usage', 
            'kwh', 'therms', 'gas', 'electric', 'pge', 'utility', 'monthly'
        ])
        
        if is_bill_analysis:
            prompt = f"""You are analyzing utility bills and energy usage data. The context contains information extracted from PDF utility bills. The text may contain some formatting artifacts from PDF extraction, but focus on identifying key billing information.

                Context:
                {context}

                Question: {question}

                Instructions for utility bill analysis:
                - Look for dollar amounts (often marked with $ symbol)
                - Identify billing periods, dates, and months
                - Extract usage amounts (kWh for electricity, therms for gas)
                - Find account numbers, meter readings, and service addresses
                - Look for charges broken down by category (delivery, generation, taxes, etc.)
                - If text appears garbled, try to identify patterns that look like monetary amounts or usage figures
                - Pay attention to table-like structures that might contain billing data
                - Be specific about which bills/months you're referencing
                - If you can identify some information but not all, explain what you found and what might be missing

                Answer:"""
        else:
            prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. Use only the information given in the context to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

                Context:
                {context}

                Question: {question}

                Instructions:
                - Answer based only on the provided context
                - If the context doesn't contain relevant information, say "I don't have enough information in the provided context to answer this question."
                - Be specific and cite which source(s) you're referencing when possible
                - Provide a clear, concise answer

                Answer:"""
        
        return prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        vector_info = self.vector_store.get_collection_info()
        
        return {
            'vector_store': vector_info,
            'model': self.model,
            'chunk_size': self.doc_processor.chunk_size,
            'chunk_overlap': self.doc_processor.chunk_overlap,
            'use_ocr': self.doc_processor.use_ocr
        }