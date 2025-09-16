import sys
import importlib
from pathlib import Path

# Remove any cached modules
modules_to_remove = [name for name in sys.modules if 'document_processor' in name]
for module in modules_to_remove:
    del sys.modules[module]

# Add src to path
sys.path.insert(0, 'src')

# Import fresh
from document_processor import DocumentProcessor

# Test
dp = DocumentProcessor(use_ocr=True)
file_path = Path('./documents/pge-may-2025.pdf')

print("Testing with fresh import:")
print(f"Is utility bill: {dp._is_utility_bill(file_path.name)}")

# Test main extraction with verbose output
result = dp._extract_pdf(file_path)
print(f"Result length: {len(result)}")
print(f"Result preview: {result[:200]}")