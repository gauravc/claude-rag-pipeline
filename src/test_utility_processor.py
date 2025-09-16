#!/usr/bin/env python3
"""
Test script to verify the utility bill processor is working correctly.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from utility_bill_processor import UtilityBillProcessor

def test_utility_processor(file_path: str):
    """Test the utility bill processor on a specific file."""
    
    print(f"Testing utility bill processor on: {file_path}")
    print("=" * 60)
    
    # Check if file exists
    pdf_file = Path(file_path)
    if not pdf_file.exists():
        print(f"Error: File {file_path} not found")
        return
    
    # Initialize processor
    try:
        processor = UtilityBillProcessor()
        print("✓ Utility bill processor initialized")
    except Exception as e:
        print(f"✗ Failed to initialize processor: {e}")
        return
    
    # Process the bill
    try:
        print("Processing utility bill...")
        result = processor.process_utility_bill(pdf_file)
        
        print("\n" + "=" * 60)
        print("EXTRACTED INFORMATION:")
        print("=" * 60)
        
        if result and result.strip():
            print(result)
            
            # Save to file
            output_file = f"utility_test_output_{pdf_file.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n✓ Full output saved to: {output_file}")
            
        else:
            print("✗ No information extracted")
            
    except Exception as e:
        print(f"✗ Error processing bill: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_utility_processor.py <path_to_pdf>")
        print("Example: python test_utility_processor.py ./documents/pge-may-2025.pdf")
        sys.exit(1)
    
    test_utility_processor(sys.argv[1])