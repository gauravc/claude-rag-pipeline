# src/evaluate.py
from rag_pipeline import RAGPipeline
import json

def evaluate_pipeline():
    pipeline = RAGPipeline()
    
    # Define test questions and expected information
    test_cases = [
        {
            "question": "What are the main topics discussed?",
            "expected_sources": ["document1.pdf"]
        }
        # Add more test cases
    ]
    
    results = []
    for case in test_cases:
        result = pipeline.query(case["question"])
        results.append({
            "question": case["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "expected_sources": case["expected_sources"]
        })
    
    return results