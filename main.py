import os
import re
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
import pytesseract
import fitz  # PyMuPDF
import pdfplumber
import argparse
from PIL import Image
import logging
import tempfile
import shutil
import atexit
import time
import gc
from typing import List, Dict, Any, Tuple
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger("table_detector")

# Create a temporary directory for processing
TEMP_DIR = tempfile.mkdtemp()

def cleanup_temp_dir():
    """Clean up temporary directory with retries and delays."""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            gc.collect()
            files = os.listdir(TEMP_DIR)
            
            for file in files:
                file_path = os.path.join(TEMP_DIR, file)
                try:
                    if os.path.exists(file_path):
                        os.chmod(file_path, 0o777)
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not remove file {file}: {str(e)}")
            
            time.sleep(retry_delay)
            
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR, ignore_errors=True)
            
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Cleanup attempt {attempt + 1} failed: {str(e)}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to clean up temporary directory after {max_retries} attempts: {str(e)}")

# Register cleanup function
atexit.register(cleanup_temp_dir)

# Known title patterns for financial statements
TABLE_PATTERNS = {
    "Balance Sheet": re.compile(r"(?:\b|^)(?:consolidated\s+)?balance\s+sheet(?:\s+as\s+(?:at|of|on))?", re.IGNORECASE),
    "Profit and Loss": re.compile(r"(?:\b|^)(?:consolidated\s+)?(?:statement\s+of\s+)?profit\s+and\s+loss|income\s+statement", re.IGNORECASE),
    "Cash Flow": re.compile(r"(?:\b|^)(?:consolidated\s+)?(?:statement\s+of\s+)?cash\s+flows?", re.IGNORECASE),
    "Liabilities": re.compile(r"(?:\b|^)(?:current|non-current|total)?\s*liabilities(?:\b|$)", re.IGNORECASE),
    "Expenses": re.compile(r"(?:\b|^)(?:operating\s+)?expenses(?:\b|$)", re.IGNORECASE),
    "Notes": re.compile(r"(?:\b|^)notes(?:\s+to|\s+on)?\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements?", re.IGNORECASE),
    "Capital Account": re.compile(r"(?:\b|^)capital\s+account(?:\b|$)", re.IGNORECASE),
    "Share Capital": re.compile(r"(?:\b|^)(?:share|shareholders['']?\s+)?capital(?:\b|$)", re.IGNORECASE),
    "Loans": re.compile(r"(?:\b|^)(?:secured|unsecured)?\s*(?:term|long\s+term|short\s+term)?\s*loans?(?:\b|$)", re.IGNORECASE),
    "Financial Statements": re.compile(r"(?:\b|^)(?:consolidated\s+)?financial\s+statements?(?:\b|$)", re.IGNORECASE),
    "Surplus": re.compile(r"(?:\b|^)(?:capital|revenue|general|other)?\s*surplus(?:\b|$)", re.IGNORECASE)
}

# Patterns to exclude from title search
EXCLUDE_PATTERNS = {
    "Net Profit": re.compile(r"(?:\b|^)net\s+profit(?:\b|$)", re.IGNORECASE),
    "Current Asset": re.compile(r"(?:\b|^)current\s+asset(?:\b|$)", re.IGNORECASE),
    "Estimates": re.compile(r"(?:\b|^)estimates(?:\b|$)", re.IGNORECASE),
    "Report": re.compile(r"(?:\b|^)report(?:\b|$)", re.IGNORECASE),
    "Responsibilities": re.compile(r"(?:\b|^)responsibilities(?:\b|$)", re.IGNORECASE)
}

def clean_text(text: str) -> str:
    """Clean text for better pattern matching."""
    if not isinstance(text, str):
        return ""
    
    # Remove hyphenation at line breaks
    text = re.sub(r'-\s*\n', '', text)
    # Replace newlines with spaces
    text = re.sub(r'\n', ' ', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return text.strip()

def detect_tables_in_image(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect tables in an image using OpenCV with optimized parameters."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding with optimized parameters
    binary = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,  # Increased block size for better line detection
        5    # Increased constant for better contrast
    )
    
    # Create kernels for line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))  # Increased width
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))    # Increased height
    
    # Detect horizontal lines with improved parameters
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)
    
    # Detect vertical lines with improved parameters
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
    
    # Combine horizontal and vertical lines
    table_cells = cv2.add(horizontal_lines, vertical_lines)
    
    # Clean up the combined image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_cells = cv2.morphologyEx(table_cells, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours with improved parameters
    contours, _ = cv2.findContours(
        table_cells,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter contours based on improved criteria
    table_bboxes = []
    min_area = 15000  # Increased minimum area
    max_area = image.shape[0] * image.shape[1] * 0.8  # Maximum 80% of image area
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h)
        
        # Improved filtering criteria
        if (min_area < area < max_area and  # Area check
            0.2 < aspect_ratio < 5 and       # Aspect ratio check
            w > 100 and h > 100):           # Minimum dimensions
            
            # Calculate line density
            roi = binary[y:y+h, x:x+w]
            line_density = np.sum(roi == 255) / (w * h)
            
            # Only include if there's sufficient line density
            if line_density > 0.05:  # At least 5% of the area should be lines
                # Expand the bounding box with padding
                padding_x = int(w * 0.05)  # 5% padding
                padding_y = int(h * 0.05)
                
                x = max(0, x - padding_x)
                y = max(0, y - padding_y - 50)  # Extra space above for title
                w = min(image.shape[1] - x, w + 2 * padding_x)
                h = min(image.shape[0] - y, h + 2 * padding_y)
                
                table_bboxes.append((x, y, x + w, y + h))
    
    return table_bboxes

def boxes_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
    """Check if two bounding boxes overlap."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    return x1 < x2 and y1 < y2

def extract_text_with_tesseract(image: np.ndarray) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """Extract text from image using Tesseract OCR."""
    try:
        # Run OCR with LSTM engine
        ocr_data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config='--psm 6 --oem 3'
        )
        
        # Process OCR results
        text_elements = []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 60:  # Only consider high confidence results
                text = ocr_data['text'][i]
                if text.strip():
                    box = (
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['left'][i] + ocr_data['width'][i],
                        ocr_data['top'][i] + ocr_data['height'][i]
                    )
                    text_elements.append((text, box))
        
        return text_elements
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {str(e)}")
        return []

def find_nearest_title(table_bbox: Tuple[float, float, float, float], text_elements: List[Tuple[str, Tuple[float, float, float, float]]], y_threshold: float = 500) -> str:
    """Find the closest title text above the table."""
    if not table_bbox or not text_elements:
        return None
    
    table_x0, table_y0, table_x1, table_y1 = table_bbox
    closest_title = None
    min_dist = float('inf')

    # Sort text elements by y-coordinate (top to bottom)
    sorted_elements = sorted(text_elements, key=lambda x: (x[1][1] + x[1][3])/2)
    
    # First pass: Look for exact matches
    for text, box in sorted_elements:
        x0, y0, x1, y1 = box
        mid_y = (y0 + y1) / 2
        mid_x = (x0 + x1) / 2
        
        # Look for titles above the table
        if mid_y < table_y0:
            # Check if text is horizontally aligned with table
            if (x0 <= table_x1 and x1 >= table_x0) or \
               (abs(mid_x - (table_x0 + table_x1)/2) < (table_x1 - table_x0)):
                
                for title, pattern in TABLE_PATTERNS.items():
                    if pattern.search(text):
                        dist = table_y0 - mid_y
                        if dist < min_dist and dist < y_threshold:
                            min_dist = dist
                            closest_title = title
                            break
    
    # Second pass: If no exact match found, look for partial matches
    if not closest_title:
        for text, box in sorted_elements:
            x0, y0, x1, y1 = box
            mid_y = (y0 + y1) / 2
            mid_x = (x0 + x1) / 2
            
            if mid_y < table_y0:
                # More lenient horizontal alignment check
                if (x0 <= table_x1 + 100 and x1 >= table_x0 - 100) or \
                   (abs(mid_x - (table_x0 + table_x1)/2) < (table_x1 - table_x0) * 1.5):
                    
                    text_lower = text.lower()
                    for title, pattern in TABLE_PATTERNS.items():
                        pattern_words = set(re.findall(r'\b\w+\b', pattern.pattern.lower()))
                        text_words = set(re.findall(r'\b\w+\b', text_lower))
                        common_words = pattern_words.intersection(text_words)
                        
                        if len(common_words) >= 2:  # At least 2 significant words match
                            dist = table_y0 - mid_y
                            if dist < min_dist and dist < y_threshold:
                                min_dist = dist
                                closest_title = title
                                break

    return closest_title

def extract_matching_line(text: str, pattern: re.Pattern) -> str:
    """Extract the whole line containing a pattern match if it's less than 12 words."""
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if pattern.search(line):
            # Check for excluded patterns
            for exclude_name, exclude_pattern in EXCLUDE_PATTERNS.items():
                if exclude_pattern.search(line):
                    logger.info(f"Excluding line due to pattern '{exclude_name}': '{line}'")
                    return None
            
            # Count words in the line
            word_count = len(line.split())
            if word_count <= 12 and len(line) > 3:  # Added length check
                return line
    return None

def select_best_title(titles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the best title from multiple candidates for a table."""
    if not titles:
        return None
    
    # Sort by detection method priority and position
    # Priority: PyMuPDF > PDFPlumber > OCR
    method_priority = {
        "PyMuPDF": 1,
        "PDFPlumber": 2,
        "OCR": 3
    }
    
    # Sort titles by priority and position
    sorted_titles = sorted(
        titles,
        key=lambda x: (
            method_priority.get(x.get("Method", "OCR"), 3),
            x.get("Position", float('inf'))
        )
    )
    
    # Return only the essential fields
    best_title = sorted_titles[0]
    return {
        "Table Title": best_title["Table Title"],
        "Page Number": best_title["Page Number"]
    }

def select_best_title_for_page(titles: List[str]) -> str:
    """Select the best title from all titles found on a page."""
    if not titles:
        return None
    
    # Define pattern priorities (higher number = higher priority)
    pattern_priorities = {
        "Balance Sheet": 10,
        "Profit and Loss": 9,
        "Cash Flow": 8,
        "Cash and Cash Equivalents": 7,
        "Liabilities": 6,
        "Expenses": 4,
        "Notes": 3,
        "Capital Account": 2
    }
    
    # Score each title
    scored_titles = []
    for title in titles:
        score = 0
        title_lower = title.lower()
        
        # Check for exact matches first
        for pattern_name, pattern in TABLE_PATTERNS.items():
            if pattern.search(title):
                score = pattern_priorities.get(pattern_name, 0) * 2
                break
        
        # If no exact match, check for partial matches
        if score == 0:
            for pattern_name, pattern in TABLE_PATTERNS.items():
                pattern_words = set(re.findall(r'\b\w+\b', pattern.pattern.lower()))
                title_words = set(re.findall(r'\b\w+\b', title_lower))
                common_words = pattern_words.intersection(title_words)
                
                if len(common_words) >= 2:
                    score = pattern_priorities.get(pattern_name, 0)
                    break
        
        # Add length bonus (prefer shorter titles)
        score += (20 - min(len(title.split()), 20))
        
        scored_titles.append((title, score))
    
    # Sort by score (highest first) and return the best title
    if scored_titles:
        scored_titles.sort(key=lambda x: x[1], reverse=True)
        return scored_titles[0][0]
    
    return None

def extract_from_digital_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tables and potential titles from text-based PDFs."""
    results = []
    table_titles = {}  # Dictionary to store titles by page and position
    
    # First attempt with fitz (PyMuPDF) for text-level detection
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                page_text = page.get_text()
                page_num = i + 1
                
                # Match standard financial table patterns
                for title, pattern in TABLE_PATTERNS.items():
                    matching_line = extract_matching_line(page_text, pattern)
                    if matching_line and len(matching_line) > 3:  # Added length check
                        key = (page_num, matching_line)
                        if key not in table_titles:
                            table_titles[key] = []
                        table_titles[key].append({
                            "Table Title": matching_line,
                            "Page Number": page_num,
                            "Method": "PyMuPDF",
                            "Position": page_text.find(matching_line)
                        })
    except Exception as e:
        logger.warning(f"Error in PyMuPDF extraction: {e}")
    
    # Additional attempt with pdfplumber for structural detection
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                
                # Extract tables structurally
                tables = page.extract_tables()
                if tables:
                    # Check table headers for known financial terms
                    for table_idx, table in enumerate(tables):
                        if not table or len(table) < 2:  # Skip empty tables
                            continue
                        
                        # Check first few rows for title matches
                        rows_to_check = min(3, len(table))
                        for row_idx in range(rows_to_check):
                            row = table[row_idx]
                            row_text = ' '.join(str(cell) for cell in row if cell)
                            if len(row_text) > 3:  # Added length check
                                for title, pattern in TABLE_PATTERNS.items():
                                    matching_line = extract_matching_line(row_text, pattern)
                                    if matching_line:
                                        key = (page_num, matching_line)
                                        if key not in table_titles:
                                            table_titles[key] = []
                                        table_titles[key].append({
                                            "Table Title": matching_line,
                                            "Page Number": page_num,
                                            "Method": "PDFPlumber",
                                            "Position": row_idx
                                        })
    except Exception as e:
        logger.warning(f"Error in pdfplumber extraction: {e}")
    
    # Select best title for each table
    for titles in table_titles.values():
        best_title = select_best_title(titles)
        if best_title:
            results.append(best_title)
    
    return results

def filter_valid_titles(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out titles that are too short, invalid, or have single spaces between characters."""
    valid_results = []
    
    # Define exceptions that should always be included
    EXCEPTION_PATTERNS = {
        "Current Expenses": re.compile(r"(?:\b|^)current\s+expenses?(?:\b|$)", re.IGNORECASE),
        "Other Expenses": re.compile(r"(?:\b|^)other\s+expenses?(?:\b|$)", re.IGNORECASE),
        "Surplus": re.compile(r"(?:\b|^)(?:capital|revenue|general|other)?\s*surplus(?:\b|$)", re.IGNORECASE)
    }
    
    for result in results:
        title = result["Table Title"]
        # Remove special characters and extra spaces
        cleaned_title = re.sub(r'[^\w\s]', ' ', title)
        cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
        
        # Check if title starts with capital letter
        if not cleaned_title[0].isupper():
            logger.warning(f"Removing title that doesn't start with capital letter: '{title}'")
            continue
        
        # Check for exceptions first
        is_exception = False
        for exception_name, pattern in EXCEPTION_PATTERNS.items():
            if pattern.search(cleaned_title):
                valid_results.append(result)
                logger.info(f"Keeping exception title: '{title}' (matched pattern: {exception_name})")
                is_exception = True
                break
        
        if is_exception:
            continue
        
        # Split into words and check word count
        words = cleaned_title.split()
        word_count = len(words)
        
        # Check word count criteria
        if word_count < 3:
            logger.warning(f"Removing title with too few words: '{title}' (word count: {word_count})")
            continue
        if word_count > 12:
            logger.warning(f"Removing title with too many words: '{title}' (word count: {word_count})")
            continue
            
        # Check if words are properly spaced (not just single characters)
        has_single_char_words = any(len(word) == 1 for word in words)
        if has_single_char_words:
            logger.warning(f"Removing title with single-character words: '{title}'")
            continue
            
        # Check if words are properly formed (not just single letters with spaces)
        if all(len(word) == 1 for word in words):
            logger.warning(f"Removing title with only single letters: '{title}'")
            continue
            
        valid_results.append(result)
        logger.info(f"Keeping valid title: '{title}' (word count: {word_count})")
    
    return valid_results

def extract_table_titles(pdf_path: str, output_csv: str) -> None:
    """Extract table titles from PDF using multiple detection methods."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Check if PDF is scanned or text-based
    is_scanned = is_scanned_pdf(pdf_path)
    logger.info(f"PDF type: {'Scanned' if is_scanned else 'Digital'}")
    
    if is_scanned:
        logger.info("Detected scanned PDF - using image-based detection")
        try:
            # Convert PDF to images with higher DPI for better quality
            images = convert_from_path(
                pdf_path,
                dpi=400,  # Increased DPI for better quality
                output_folder=TEMP_DIR,
                fmt='png',
                thread_count=4,
                use_pdftocairo=True,
                grayscale=False
            )
            
            results = []

            # Process each page
            for page_num, image in enumerate(images, start=1):
                logger.info(f"Processing page {page_num}")
                
                # Convert PIL Image to OpenCV format
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Detect tables in the image
                table_bboxes = detect_tables_in_image(cv_image)
                logger.info(f"Found {len(table_bboxes)} tables on page {page_num}")
                
                # Process only the first table on the page
                if table_bboxes:
                    table_bbox = table_bboxes[0]  # Take only the first table
                    logger.info(f"Processing table at position: {table_bbox}")
                    
                    # Extract title using OCR
                    title = extract_table_title_with_ocr(cv_image, table_bbox)
                    if title:
                        logger.info(f"Found title: '{title}'")
                        # Check if this is a valid financial table title
                        for pattern_name, pattern in TABLE_PATTERNS.items():
                            if pattern.search(title):
                                results.append({
                                    "Table Title": title,
                                    "Page Number": page_num
                                })
                                logger.info(f"Added valid title on page {page_num}: {title}")
                                break
                        else:
                            logger.warning(f"No title found for table on page {page_num}")
                else:
                    logger.warning(f"No tables found on page {page_num}")
                
                # Clean up
                image.close()
                del image
                gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing scanned PDF: {str(e)}")
            results = []
    else:
        logger.info("Detected text-based PDF - using direct text extraction")
        try:
            results = extract_from_digital_pdf(pdf_path)
            logger.info(f"Found {len(results)} titles in digital PDF")
        except Exception as e:
            logger.error(f"Error processing digital PDF: {str(e)}")
            results = []
    
    # Filter out short titles
    results = filter_valid_titles(results)
    logger.info(f"After filtering, found {len(results)} valid titles")
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        logger.info(f"Extracted {len(df)} unique table titles from {pdf_path}")
        logger.info(f"Results saved to: {output_csv}")
    else:
        logger.warning("No tables with titles were found in the document")

def is_scanned_pdf(pdf_path: str) -> bool:
    """Determine if a PDF is scanned (image-based) or digital (text-based)."""
    try:
        with fitz.open(pdf_path) as doc:
            # Sample up to 3 pages to determine if it's scanned
            pages_to_check = min(3, len(doc))
            total_text = ""
            
            for i in range(pages_to_check):
                page = doc[i]
                total_text += page.get_text().strip()
                
                # If we found substantial text, it's likely a text PDF
                if len(total_text) > 300:
                    logger.debug(f"Found {len(total_text)} chars of text - considering as digital PDF")
                    return False
            
            # If we've checked pages but found little text, likely a scanned PDF
            if len(total_text) < 100:
                logger.debug(f"Found only {len(total_text)} chars of text - considering as scanned PDF")
                return True
            
            # In between - check if there are images
            image_count = 0
            for i in range(pages_to_check):
                page = doc[i]
                images = page.get_images()
                image_count += len(list(images))
            
            # If many images but little text, likely scanned
            return image_count > 0 and len(total_text) < 200
    
    except Exception as e:
        logger.warning(f"Error checking if PDF is scanned: {e}")
        return False

def extract_table_title_with_ocr(image: np.ndarray, table_bbox: Tuple[int, int, int, int]) -> str:
    """Extract table title using OCR from the region above the table."""
    x0, y0, x1, y1 = table_bbox
    logger.info(f"Searching for title in region: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
    
    # Define the search area above the table
    search_height = 100  # pixels above table to search for title
    search_y0 = max(0, y0 - search_height)
    search_y1 = y0
    
    # Extract the region above the table
    search_region = image[search_y0:search_y1, x0:x1]
    
    if search_region.size == 0:
        logger.warning("Search region is empty")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Run OCR on the search region
    try:
        # Use Tesseract with specific configuration for title detection
        custom_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        ocr_text = pytesseract.image_to_string(binary, config=custom_config)
        
        if not ocr_text.strip():
            logger.warning("No text found in OCR result")
            return None
            
        # Split into lines and process each line
        lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
        logger.info(f"Found {len(lines)} lines in OCR result")
        
        # Check each line against our patterns
        for line in lines:
            # Clean the line
            line = re.sub(r'[^\w\s]', ' ', line)  # Remove special characters
            line = re.sub(r'\s+', ' ', line).strip()  # Normalize whitespace
            
            # Skip empty lines or very short lines
            if len(line) < 4:
                logger.debug(f"Skipping short line: '{line}' (length: {len(line)})")
                continue
                
            # Check word count
            if len(line.split()) > 12:
                logger.debug(f"Skipping long line: '{line}' (words: {len(line.split())})")
                continue
            
            # Check for excluded patterns
            should_exclude = False
            for exclude_name, exclude_pattern in EXCLUDE_PATTERNS.items():
                if exclude_pattern.search(line):
                    logger.info(f"Excluding line due to pattern '{exclude_name}': '{line}'")
                    should_exclude = True
                    break
            
            if should_exclude:
                continue
                
            logger.info(f"Processing line: '{line}'")
            
            # Check for exact matches
            for title, pattern in TABLE_PATTERNS.items():
                if pattern.search(line):
                    logger.info(f"Found exact match for pattern '{title}': '{line}'")
                    return line
            
            # Check for partial matches
            line_lower = line.lower()
            for title, pattern in TABLE_PATTERNS.items():
                pattern_words = set(re.findall(r'\b\w+\b', pattern.pattern.lower()))
                line_words = set(re.findall(r'\b\w+\b', line_lower))
                common_words = pattern_words.intersection(line_words)
                
                if len(common_words) >= 2:
                    logger.info(f"Found partial match for pattern '{title}': '{line}' (common words: {common_words})")
                    return line
        
        logger.warning("No matching title found in OCR text")
        return None
        
    except Exception as e:
        logger.error(f"OCR title extraction failed: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract table titles from PDF documents')
    parser.add_argument('input_file', help='Path to the input PDF file')
    parser.add_argument('--output', '-o', required=True,
                      help='Path to the output CSV file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        logger.error(f"Input file '{args.input_file}' does not exist!")
    else:
        extract_table_titles(args.input_file, args.output)
