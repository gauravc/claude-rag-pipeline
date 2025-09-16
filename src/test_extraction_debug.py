import sys
sys.path.append('src')
from document_processor import DocumentProcessor
from pathlib import Path

dp = DocumentProcessor(use_ocr=True)
file_path = Path('./documents/pge-may-2025.pdf')

print("=== TESTING EXTRACTION METHODS ===")
print(f"File: {file_path.name}")
print(f"Is utility bill: {dp._is_utility_bill(file_path.name)}")
print(f"OCR enabled: {dp.use_ocr}")
print()

# Test OCR directly
print("1. Testing OCR method directly:")
try:
    ocr_result = dp._extract_pdf_ocr(file_path)
    print(f"   OCR length: {len(ocr_result)}")
    print(f"   OCR preview: {ocr_result[:200]}")
except Exception as e:
    print(f"   OCR failed: {e}")

print()

# Test the main extraction method with verbose output
print("2. Testing main _extract_pdf method:")
try:
    main_result = dp._extract_pdf(file_path)
    print(f"   Main method length: {len(main_result)}")
    print(f"   Main method preview: {main_result[:200]}")
except Exception as e:
    print(f"   Main method failed: {e}")