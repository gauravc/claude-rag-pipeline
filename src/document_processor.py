import os
import re
from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
from docx import Document
import tiktoken
import cv2
import numpy as np

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, use_ocr: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ocr = use_ocr
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Configure Tesseract path if needed (uncomment and adjust for your system)
        # pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # macOS with Homebrew
        # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux
    
    def load_documents(self, directory: str) -> List[Dict[str, Any]]:
        """Load documents from a directory."""
        documents = []
        directory_path = Path(directory)
        
        for file_path in directory_path.glob("**/*"):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == '.pdf':
                        print(f"Processing PDF: {file_path.name}")
                        content = self._extract_pdf(file_path)
                    elif file_path.suffix.lower() in ['.docx', '.doc']:
                        print(f"Processing DOCX: {file_path.name}")
                        content = self._extract_docx(file_path)
                    elif file_path.suffix.lower() == '.txt':
                        print(f"Processing TXT: {file_path.name}")
                        content = self._extract_text(file_path)
                    else:
                        continue
                    
                    if content and len(content.strip()) > 0:
                        documents.append({
                            'content': content,
                            'source': str(file_path),
                            'filename': file_path.name
                        })
                        print(f"✓ Extracted {len(content)} characters from {file_path.name}")
                    else:
                        print(f"✗ No content extracted from {file_path.name}")
                        
                except Exception as e:
                    print(f"✗ Error processing {file_path}: {e}")
        
        return documents
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files with utility bill optimization."""
        
        try:
            # For utility bills, prioritize OCR since standard methods fail
            if self._is_utility_bill(file_path.name):
                print(f"  Utility bill detected - using OCR extraction")
                if self.use_ocr:
                    print(f"  Running OCR...")
                    ocr_text = self._extract_pdf_ocr(file_path)
                    print(f"  OCR extracted {len(ocr_text)} characters")
                    
                    if len(ocr_text.strip()) > 50:  # OCR produced results
                        print(f"  ✓ OCR successful, using OCR results")
                        cleaned_text = self._clean_text(ocr_text)
                        print(f"  ✓ After cleaning: {len(cleaned_text)} characters")
                        return cleaned_text
                    else:
                        print(f"  ✗ OCR produced minimal text ({len(ocr_text)} chars), trying other methods")
                else:
                    print(f"  OCR disabled, trying other methods")
            else:
                print(f"  Regular PDF - trying standard methods")
            
            # For regular PDFs or if utility OCR failed, try standard methods
            print(f"  Trying pdfplumber extraction...")
            text = self._extract_pdf_pdfplumber(file_path)
            
            # If pdfplumber gets good results, use it
            if len(text.strip()) > 200:
                print(f"  ✓ pdfplumber extracted {len(text)} characters")
                return self._clean_text(text)
            
            # Otherwise try PyMuPDF
            print(f"  Trying PyMuPDF extraction...")
            pymupdf_text = self._extract_pdf_pymupdf(file_path)
            
            if len(pymupdf_text.strip()) > len(text.strip()):
                text = pymupdf_text
                print(f"  ✓ PyMuPDF extracted {len(text)} characters")
            
            # Last resort: try OCR if we haven't already and have poor results
            if len(text.strip()) < 100 and self.use_ocr and not self._is_utility_bill(file_path.name):
                print(f"  Poor extraction results, trying OCR as last resort...")
                ocr_text = self._extract_pdf_ocr(file_path)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    print(f"  ✓ OCR extracted {len(text)} characters")
            
            return self._clean_text(text)
        
        except Exception as e:
            print(f"  Error extracting from {file_path}: {e}")
            return ""
    
    def _extract_pdf_pymupdf(self, file_path: Path) -> str:
        """Extract text using PyMuPDF (good for most PDFs)."""
        doc = fitz.open(file_path)
        text = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Try multiple text extraction methods
            page_text = ""
            
            # Method 1: Standard text extraction
            standard_text = page.get_text()
            if standard_text and len(standard_text.strip()) > 50:
                page_text = standard_text
            else:
                # Method 2: Extract with layout preservation
                layout_text = page.get_text("dict")
                page_text = self._extract_from_layout(layout_text)
            
            # Method 3: Extract tables separately
            tables = page.find_tables()
            table_text = ""
            for table in tables:
                try:
                    table_data = table.extract()
                    for row in table_data:
                        if row:
                            table_text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                except Exception as e:
                    continue
            
            # Combine text sources
            combined_text = page_text + "\n" + table_text
            text += f"--- Page {page_num + 1} ---\n{combined_text}\n"
        
        doc.close()
        return text
    
    def _extract_from_layout(self, layout_dict: dict) -> str:
        """Extract text from PyMuPDF layout dictionary."""
        text = ""
        
        def extract_blocks(blocks):
            result = ""
            for block in blocks:
                if block.get('type') == 0:  # Text block
                    for line in block.get('lines', []):
                        line_text = ""
                        for span in line.get('spans', []):
                            span_text = span.get('text', '')
                            if span_text.strip():
                                line_text += span_text + " "
                        if line_text.strip():
                            result += line_text.strip() + "\n"
            return result
        
        if 'blocks' in layout_dict:
            text = extract_blocks(layout_dict['blocks'])
        
        return text
    
    def _extract_pdf_pdfplumber(self, file_path: Path) -> str:
        """Extract text using pdfplumber optimized for utility bills."""
        text = ""
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = f"--- Page {page_num + 1} ---\n"
                
                # Method 1: Try structured table extraction first
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text", 
                        "snap_tolerance": 5,
                        "join_tolerance": 5,
                        "edge_min_length": 10,
                        "min_words_vertical": 1,
                        "min_words_horizontal": 1,
                    }
                )
                
                # Extract and format tables
                for i, table in enumerate(tables):
                    if table and len(table) > 0:
                        page_text += f"\n=== TABLE {i+1} ===\n"
                        for row in table:
                            if row and any(cell and str(cell).strip() for cell in row):
                                clean_row = []
                                for cell in row:
                                    if cell:
                                        clean_cell = str(cell).strip()
                                        # Clean common OCR artifacts
                                        clean_cell = self._clean_cell_text(clean_cell)
                                        if clean_cell:
                                            clean_row.append(clean_cell)
                                if clean_row:
                                    page_text += " | ".join(clean_row) + "\n"
                
                # Method 2: Extract text with better character filtering
                try:
                    extracted_text = page.extract_text(
                        x_tolerance=2,
                        y_tolerance=2,
                        layout=False,
                        x_density=7.25,
                        y_density=13
                    )
                    
                    if extracted_text:
                        # Clean the extracted text
                        cleaned_text = self._clean_extracted_text(extracted_text)
                        if len(cleaned_text.strip()) > 50:  # Only add if substantial
                            page_text += f"\n=== TEXT CONTENT ===\n{cleaned_text}\n"
                
                except Exception as e:
                    print(f"    Text extraction failed for page {page_num + 1}: {e}")
                
                # Method 3: Try character-level extraction for better accuracy
                chars = page.chars
                if chars:
                    char_text = self._extract_from_chars(chars)
                    if len(char_text.strip()) > len(extracted_text or ""):
                        page_text += f"\n=== CHAR EXTRACTION ===\n{char_text}\n"
                
                text += page_text + "\n"
        
        return text
    
    def _clean_cell_text(self, text: str) -> str:
        """Clean individual table cell text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR issues in utility bills
        replacements = {
            'O': '0',  # O -> 0 in numbers
            'l': '1',  # l -> 1 in numbers  
            'S': '5',  # S -> 5 in numbers
        }
        
        # Only apply replacements if the cell looks like it should contain numbers
        if any(char.isdigit() for char in text):
            import re
            # Replace O with 0 in contexts that look like numbers
            text = re.sub(r'\b\d*O\d*\b', lambda m: m.group().replace('O', '0'), text)
            text = re.sub(r'\b\d*l\d*\b', lambda m: m.group().replace('l', '1'), text)
        
        return text
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text with utility bill specific cleaning."""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that are mostly artifacts
            if len(line.strip()) < 3:
                continue
            
            # Skip lines with too many special characters (likely OCR artifacts)
            special_chars = sum(1 for c in line if not c.isalnum() and c not in ' .,/$-():')
            if special_chars > len(line) * 0.5:
                continue
            
            # Clean the line
            cleaned_line = line.strip()
            
            # Try to preserve important patterns
            import re
            # Preserve dollar amounts, dates, account numbers
            if re.search(r'[\$\d/.-]', cleaned_line):
                cleaned_lines.append(cleaned_line)
            elif any(word in cleaned_line.lower() for word in ['pge', 'pg&e', 'account', 'total', 'kwh', 'therm']):
                cleaned_lines.append(cleaned_line)
            elif len(cleaned_line) > 10 and any(c.isalpha() for c in cleaned_line):
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_from_chars(self, chars) -> str:
        """Extract text from character-level data."""
        if not chars:
            return ""
        
        # Group characters by line
        lines = {}
        for char in chars:
            y = round(char['top'], 1)  # Round to group by line
            if y not in lines:
                lines[y] = []
            lines[y].append(char)
        
        # Sort lines by y-position and characters by x-position
        text_lines = []
        for y in sorted(lines.keys(), reverse=True):  # Top to bottom
            line_chars = sorted(lines[y], key=lambda c: c['x0'])  # Left to right
            line_text = ''.join(char['text'] for char in line_chars)
            if line_text.strip():
                text_lines.append(line_text.strip())
        
        return '\n'.join(text_lines)
    
    def _extract_pdf_ocr(self, file_path: Path) -> str:
        """Extract text using OCR for image-heavy PDFs (utility bills often need this)."""
        doc = fitz.open(file_path)
        text = ""
        
        for page_num in range(min(doc.page_count, 5)):  # Process up to 5 pages for bills
            page = doc[page_num]
            
            # Convert page to high-resolution image for better OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))  # 3x zoom for better OCR
            img_data = pix.tobytes("png")
            
            # Perform OCR with better configuration for utility bills
            try:
                img = Image.open(io.BytesIO(img_data))
                
                # Use custom OCR configuration for better number recognition
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,$/()-:@# '
                
                page_text = pytesseract.image_to_string(
                    img, 
                    lang='eng',
                    config=custom_config
                )
                
                if page_text.strip():
                    text += f"--- OCR Page {page_num + 1} ---\n{page_text}\n"
                    
            except Exception as e:
                print(f"    OCR failed for page {page_num + 1}: {e}")
                continue
        
        doc.close()
        return text
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX files."""
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return self._clean_text(text)
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text from TXT files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text, especially for utility bills."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR/extraction issues
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\|\n\$\@\#\%\&\*\+\=\/\\]', '', text)
        
        # Preserve important patterns for utility bills
        # Keep dollar amounts: $123.45
        # Keep dates: 01/15/2025, Jan 15, 2025
        # Keep account numbers and meter readings
        # Keep kWh, therms, etc.
        
        # Fix common character replacement issues
        replacements = {
            "'": "'",  # Smart quote
            '"': '"',  # Smart quote open
            '"': '"',  # Smart quote close
            '•': '•',  # Bullet point
            ' ': ' ',  # Non-breaking space
            '−': '-',  # Minus sign
            '–': '-',  # En dash
            '—': '-',  # Em dash
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove excessive spacing but preserve structure
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' {3,}', '   ', text)  # Keep some spacing for alignment
        
        return text
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks with utility bill optimization."""
        chunks = []
        
        for doc in documents:
            text = doc['content']
            
            # For utility bills, try to extract structured information first
            if 'pge' in doc['filename'].lower() or 'bill' in doc['filename'].lower():
                structured_info = self._extract_bill_info(text)
                if structured_info:
                    # Add structured info as a separate chunk
                    chunks.append({
                        'content': structured_info,
                        'source': doc['source'],
                        'filename': doc['filename'],
                        'chunk_id': 0,
                        'total_chunks': 1,
                        'type': 'structured_bill_info'
                    })
            
            # Regular chunking
            doc_chunks = self._split_text(text)
            
            for i, chunk in enumerate(doc_chunks):
                chunks.append({
                    'content': chunk,
                    'source': doc['source'],
                    'filename': doc['filename'],
                    'chunk_id': i + (1 if 'pge' in doc['filename'].lower() else 0),
                    'total_chunks': len(doc_chunks) + (1 if 'pge' in doc['filename'].lower() else 0),
                    'type': 'text_chunk'
                })
        
        return chunks
    
    def _extract_bill_info(self, text: str) -> str:
        """Extract structured information from utility bills."""
        import re
        
        structured_info = []
        
        # Extract dollar amounts
        dollar_amounts = re.findall(r'\$\s*\d+[.,]\d{2}', text)
        if dollar_amounts:
            structured_info.append("AMOUNTS FOUND:")
            for amount in set(dollar_amounts):  # Remove duplicates
                structured_info.append(f"  {amount}")
        
        # Extract dates
        dates = re.findall(r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\b', text)
        if dates:
            structured_info.append("\nDATES FOUND:")
            for date in set(dates):
                structured_info.append(f"  {date}")
        
        # Extract energy units
        energy_usage = re.findall(r'\d+[.,]?\d*\s*(?:kWh|kwh|KWH|therm|Therm|THERM)', text, re.IGNORECASE)
        if energy_usage:
            structured_info.append("\nENERGY USAGE:")
            for usage in set(energy_usage):
                structured_info.append(f"  {usage}")
        
        # Extract account-related info
        account_matches = re.findall(r'(?:Account|account|ACCOUNT)[\s\w]*?(\d{8,})', text)
        if account_matches:
            structured_info.append("\nACCOUNT NUMBERS:")
            for account in set(account_matches):
                structured_info.append(f"  {account}")
        
        # Extract service period
        period_matches = re.findall(r'(?:Service|service|SERVICE|Period|period|PERIOD)[\s\w]*?(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})', text)
        if period_matches:
            structured_info.append("\nSERVICE PERIODS:")
            for period in set(period_matches):
                structured_info.append(f"  {period}")
        
        # Look for bill totals
        total_matches = re.findall(r'(?:Total|total|TOTAL|Amount|amount|AMOUNT)[\s\w]*?\$\s*\d+[.,]\d{2}', text)
        if total_matches:
            structured_info.append("\nBILL TOTALS:")
            for total in set(total_matches):
                structured_info.append(f"  {total}")
        
        if structured_info:
            return "=== EXTRACTED UTILITY BILL INFORMATION ===\n" + "\n".join(structured_info)
        
        return ""
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        # Tokenize the text
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _is_utility_bill(self, filename: str) -> bool:
        """Check if a file appears to be a utility bill."""
        filename_lower = filename.lower()
        utility_indicators = [
            'pge', 'pg&e', 'pacific gas', 'electric',
            'bill', 'utility', 'energy', 'gas',
            'edison', 'sdge', 'peco', 'con ed'
        ]
        return any(indicator in filename_lower for indicator in utility_indicators)