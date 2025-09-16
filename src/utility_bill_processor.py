import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np

class UtilityBillProcessor:
    """Specialized processor for utility bills that are often scanned or image-based."""
    
    def __init__(self):
        # Configure Tesseract for better number and text recognition
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,$/():-@# '
        
        # Patterns for utility bill information
        self.patterns = {
            'amounts': r'\$\s*\d{1,3}(?:,\d{3})*\.\d{2}',
            'dates': r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\b',
            'kwh': r'\d+[.,]?\d*\s*(?:kWh|kwh|KWH)',
            'therms': r'\d+[.,]?\d*\s*(?:therm|Therm|THERM)',
            'account': r'(?:Account|Acct)[\s#]*(\d{8,})',
            'service_period': r'(?:Service|Bill)\s+(?:Period|Date)[\s:]*(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})\s*(?:to|-)?\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})?'
        }
    
    def process_utility_bill(self, file_path: Path) -> str:
        """Process a utility bill PDF with enhanced OCR."""
        print(f"Processing utility bill: {file_path.name}")
        
        doc = fitz.open(file_path)
        extracted_info = []
        
        for page_num in range(min(doc.page_count, 3)):  # Process first 3 pages
            print(f"  Processing page {page_num + 1}...")
            page = doc[page_num]
            
            # Convert to high-resolution image
            pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))  # 3x resolution
            img_data = pix.tobytes("png")
            
            # Process with enhanced OCR
            page_info = self._process_page_image(img_data, page_num + 1)
            if page_info:
                extracted_info.append(page_info)
        
        doc.close()
        
        if extracted_info:
            return "\n\n".join(extracted_info)
        else:
            return "No readable information could be extracted from this utility bill."
    
    def _process_page_image(self, img_data: bytes, page_num: int) -> str:
        """Process a single page image with multiple OCR techniques."""
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to numpy array for OpenCV processing
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        results = []
        
        # Method 1: Direct OCR on original image
        try:
            text1 = pytesseract.image_to_string(img, config=self.tesseract_config)
            info1 = self._extract_bill_info(text1, f"Page {page_num} - Direct OCR")
            if info1:
                results.append(info1)
        except Exception as e:
            print(f"    Direct OCR failed: {e}")
        
        # Method 2: Enhanced contrast OCR
        try:
            enhanced_img = self._enhance_image_for_ocr(img)
            text2 = pytesseract.image_to_string(enhanced_img, config=self.tesseract_config)
            info2 = self._extract_bill_info(text2, f"Page {page_num} - Enhanced OCR")
            if info2:
                results.append(info2)
        except Exception as e:
            print(f"    Enhanced OCR failed: {e}")
        
        # Method 3: Table detection and OCR
        try:
            table_info = self._extract_table_regions(img_cv, page_num)
            if table_info:
                results.append(table_info)
        except Exception as e:
            print(f"    Table detection failed: {e}")
        
        return "\n".join(results) if results else ""
    
    def _enhance_image_for_ocr(self, img: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR results."""
        # Convert to grayscale
        img = img.convert('L')
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        # Apply slight blur to reduce noise
        img = img.filter(ImageFilter.MedianFilter(size=3))
        
        return img
    
    def _extract_table_regions(self, img_cv: np.ndarray, page_num: int) -> str:
        """Try to detect and extract table-like regions."""
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Find contours (potential table cells)
        contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_info = []
        
        # Process larger contours that might be table cells
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Only process significant regions
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract the region
                cell_img = gray[y:y+h, x:x+w]
                
                # Convert back to PIL Image for OCR
                cell_pil = Image.fromarray(cell_img)
                
                try:
                    cell_text = pytesseract.image_to_string(cell_pil, config=self.tesseract_config)
                    if cell_text.strip():
                        # Look for important patterns in this cell
                        info = self._extract_bill_info(cell_text, f"Page {page_num} - Table Cell")
                        if info:
                            table_info.append(info)
                except:
                    continue
        
        return "\n".join(table_info) if table_info else ""
    
    def _extract_bill_info(self, text: str, source: str) -> str:
        """Extract structured information from OCR text."""
        if not text or len(text.strip()) < 10:
            return ""
        
        info_found = []
        
        # Extract amounts
        amounts = re.findall(self.patterns['amounts'], text)
        if amounts:
            # Remove duplicates and filter reasonable amounts
            unique_amounts = list(set(amounts))
            reasonable_amounts = [amt for amt in unique_amounts if self._is_reasonable_bill_amount(amt)]
            if reasonable_amounts:
                info_found.append(f"AMOUNTS: {', '.join(reasonable_amounts)}")
        
        # Extract dates
        dates = re.findall(self.patterns['dates'], text)
        if dates:
            unique_dates = list(set(dates))
            valid_dates = [date for date in unique_dates if self._is_valid_date(date)]
            if valid_dates:
                info_found.append(f"DATES: {', '.join(valid_dates)}")
        
        # Extract energy usage
        kwh_usage = re.findall(self.patterns['kwh'], text, re.IGNORECASE)
        if kwh_usage:
            info_found.append(f"ELECTRICITY USAGE: {', '.join(set(kwh_usage))}")
        
        therm_usage = re.findall(self.patterns['therms'], text, re.IGNORECASE)
        if therm_usage:
            info_found.append(f"GAS USAGE: {', '.join(set(therm_usage))}")
        
        # Extract account info
        accounts = re.findall(self.patterns['account'], text, re.IGNORECASE)
        if accounts:
            info_found.append(f"ACCOUNT: {', '.join(set(accounts))}")
        
        # Extract service periods
        service_periods = re.findall(self.patterns['service_period'], text, re.IGNORECASE)
        if service_periods:
            periods = [f"{start} to {end}" if end else start for start, end in service_periods]
            info_found.append(f"SERVICE PERIOD: {', '.join(periods)}")
        
        if info_found:
            return f"=== {source} ===\n" + "\n".join(info_found)
        
        return ""
    
    def _is_reasonable_bill_amount(self, amount_str: str) -> bool:
        """Check if an amount seems reasonable for a utility bill."""
        try:
            # Remove $ and convert to float
            amount = float(amount_str.replace('$', '').replace(',', ''))
            # Reasonable utility bill range: $10 - $2000
            return 10.0 <= amount <= 2000.0
        except:
            return False
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Check if a date string seems valid."""
        try:
            # Basic validation - should have reasonable month/day values
            parts = re.split(r'[/.-]', date_str)
            if len(parts) == 3:
                month, day, year = map(int, parts)
                return 1 <= month <= 12 and 1 <= day <= 31 and 2020 <= year <= 2030
        except:
            return False
        return False