#!/usr/bin/env python3
"""
Test script to verify the document processor integration is working.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_document_processor():
    """Test the document processor with utility bill detection."""
    
    print("Testing Document Processor Integration")
    print("=" * 50)
    
    try:
        from document_processor import DocumentProcessor
        print("✓ DocumentProcessor imported successfully")
    except Exception as e:
        print(f"✗ Failed to import DocumentProcessor: {e}")
        return False
    
    # Initialize processor
    try:
        processor = DocumentProcessor(use_ocr=True)
        print("✓ DocumentProcessor initialized")
    except Exception as e:
        print(f"✗ Failed to initialize DocumentProcessor: {e}")
        return False
    
    # Test utility bill detection
    test_files = [
        "pge-may-2025.pdf",
        "pge-jun-2025.pdf", 
        "normal-document.pdf"
    ]
    
    print("\nTesting utility bill detection:")
    for filename in test_files:
        is_utility = processor._is_utility_bill(filename)
        print(f"  {filename}: {'✓ UTILITY BILL' if is_utility else '✗ Regular document'}")
    
    # Test actual processing if file exists
    test_file = Path("./documents/pge-may-2025.pdf")
    if test_file.exists():
        print(f"\nTesting actual processing of {test_file.name}...")
        
        # Test the main load_documents method
        try:
            documents = processor.load_documents("./documents")
            print(f"✓ Loaded {len(documents)} documents")
            
            for doc in documents:
                print(f"  - {doc['filename']}: {len(doc['content'])} characters")
                
                # Show first 200 characters to see if it's clean
                preview = doc['content'][:200].replace('\n', ' ')
                print(f"    Preview: {preview}...")
                
        except Exception as e:
            print(f"✗ Failed to load documents: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    success = test_document_processor()
    if success:
        print("\n✓ Integration test passed!")
    else:
        print("\n✗ Integration test failed!")
        sys.exit(1)