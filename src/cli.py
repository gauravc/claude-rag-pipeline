#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from rag_pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description='RAG Pipeline with Claude Sonnet 4')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('--path', required=True, help='Path to documents directory')
    ingest_parser.add_argument('--clear', action='store_true', help='Clear existing documents before ingesting')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG pipeline')
    query_parser.add_argument('--question', required=True, help='Question to ask')
    query_parser.add_argument('--k', type=int, default=5, help='Number of documents to retrieve')
    query_parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for generation')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show pipeline statistics')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Debug document processing')
    debug_parser.add_argument('--file', required=True, help='Path to specific file to debug')
    debug_parser.add_argument('--method', choices=['pymupdf', 'pdfplumber', 'ocr'], 
                            help='Specific extraction method to test')
    debug_parser.add_argument('--save-text', action='store_true', 
                            help='Save extracted text to file for inspection')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all documents from vector store')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("Make sure your .env file is configured correctly with ANTHROPIC_API_KEY")
        sys.exit(1)
    
    if args.command == 'ingest':
        if not Path(args.path).exists():
            print(f"Error: Path {args.path} does not exist")
            sys.exit(1)
        
        # Clear existing documents if requested
        if args.clear:
            print("Clearing existing documents...")
            pipeline.vector_store.clear_collection()
        
        # Add verification option for debugging
        print("Starting document ingestion...")
        print("This will show what text is actually being processed and stored.")
        
        pipeline.ingest_documents(args.path)
        
        # Show statistics after ingestion
        stats = pipeline.get_stats()
        print(f"\nIngestion completed!")
        print(f"Total chunks stored: {stats['vector_store']['count']}")
        
        # Optional: Show a sample of what was stored
        if stats['vector_store']['count'] > 0:
            print("\nTesting retrieval of stored content...")
            try:
                # Test search to see what's actually stored
                test_results = pipeline.vector_store.similarity_search("PG&E bill total amount", k=2)
                if test_results:
                    print("Sample of stored content:")
                    for i, result in enumerate(test_results[:1], 1):
                        content_preview = result['content'][:300].replace('\n', ' ')
                        print(f"  Chunk {i}: {content_preview}...")
                        print(f"  Source: {result['metadata']['filename']}")
                else:
                    print("No content found in similarity search")
            except Exception as e:
                print(f"Could not test retrieval: {e}")
    
    elif args.command == 'query':
        result = pipeline.query(
            question=args.question,
            k=args.k,
            temperature=args.temperature
        )
        
        print("\n" + "="*70)
        print("ANSWER:")
        print("="*70)
        print(result['answer'])
        
        if result['sources']:
            print("\n" + "="*70)
            print("SOURCES:")
            print("="*70)
            for source in result['sources']:
                print(f"• {source}")
        
        print("\n" + "="*70)
        print(f"Retrieved {len(result['retrieved_docs'])} relevant document chunks")
        print("="*70)
    
    elif args.command == 'stats':
        stats = pipeline.get_stats()
        print("\n" + "="*70)
        print("PIPELINE STATISTICS:")
        print("="*70)
        print(f"Model: {stats['model']}")
        print(f"Documents in vector store: {stats['vector_store']['count']}")
        print(f"Chunk size: {stats['chunk_size']}")
        print(f"Chunk overlap: {stats['chunk_overlap']}")
        print(f"OCR enabled: {stats['use_ocr']}")
        print(f"Vector store path: {stats['vector_store']['db_path']}")
        print("="*70)
    
    elif args.command == 'debug':
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File {args.file} does not exist")
            sys.exit(1)
        
        print(f"Debugging extraction for: {file_path.name}")
        print("="*70)
        
        doc_processor = pipeline.doc_processor
        
        # Check if this is detected as a utility bill
        is_utility = doc_processor._is_utility_bill(file_path.name)
        print(f"Detected as utility bill: {is_utility}")
        
        if args.method:
            # Test specific method
            if args.method == 'pymupdf':
                text = doc_processor._extract_pdf_pymupdf(file_path)
            elif args.method == 'pdfplumber':
                text = doc_processor._extract_pdf_pdfplumber(file_path)
            elif args.method == 'ocr':
                text = doc_processor._extract_pdf_ocr(file_path)
            
            print(f"Method: {args.method}")
            print(f"Extracted text length: {len(text)} characters")
            print("-" * 50)
            print(text[:2000])  # Show first 2000 characters
            if len(text) > 2000:
                print(f"\n... (truncated, total length: {len(text)} characters)")
        else:
            # Test the main extraction method (which should use utility processor for bills)
            print("\nTesting main extraction method:")
            print("-" * 30)
            
            if is_utility:
                print("Using specialized utility bill processor...")
                try:
                    if hasattr(doc_processor, 'utility_processor') and doc_processor.utility_processor:
                        text = doc_processor.utility_processor.process_utility_bill(file_path)
                        print(f"Utility processor result length: {len(text)} characters")
                        print("UTILITY PROCESSOR OUTPUT:")
                        print("-" * 40)
                        print(text[:1500])
                        if len(text) > 1500:
                            print(f"\n... (truncated, total length: {len(text)} characters)")
                    else:
                        print("Utility processor not available, falling back to OCR")
                        text = doc_processor._extract_pdf_ocr(file_path)
                        print(f"OCR fallback result length: {len(text)} characters")
                        print("OCR FALLBACK OUTPUT:")
                        print("-" * 40)
                        print(text[:1500])
                except Exception as e:
                    print(f"Utility processor failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Also test standard methods for comparison
            methods = [
                ('pymupdf', doc_processor._extract_pdf_pymupdf),
                ('pdfplumber', doc_processor._extract_pdf_pdfplumber),
                ('ocr', doc_processor._extract_pdf_ocr)
            ]
            
            print(f"\nTesting standard methods for comparison:")
            for method_name, method_func in methods:
                print(f"\n{method_name.upper()}:")
                print("-" * 30)
                try:
                    text = method_func(file_path)
                    print(f"Length: {len(text)} characters")
                    print(text[:300])  # Show first 300 characters
                    if len(text) > 300:
                        print("... (truncated)")
                except Exception as e:
                    print(f"Error: {e}")
        
        if args.save_text:
            # Save full extracted text to file
            if is_utility:
                try:
                    full_text = doc_processor.utility_processor.process_utility_bill(file_path)
                    output_file = f"debug_utility_{file_path.stem}.txt"
                except:
                    full_text = doc_processor._extract_pdf(file_path)
                    output_file = f"debug_standard_{file_path.stem}.txt"
            else:
                full_text = doc_processor._extract_pdf(file_path)
                output_file = f"debug_output_{file_path.stem}.txt"
                
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"\nFull extracted text saved to: {output_file}")
    
    elif args.command == 'clear':
        print("Clearing all documents from vector store...")
        pipeline.vector_store.clear_collection()
        print("✓ Vector store cleared successfully")

if __name__ == "__main__":
    main()