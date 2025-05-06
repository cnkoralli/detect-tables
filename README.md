# Financial Statement Table Detector

A Python-based solution for detecting and extracting tables from financial statement PDFs.

## Features

- Process both digital and scanned PDF files
- Detect tables using layout analysis
- Extract table titles based on proximity and header patterns
- Generate structured output (CSV/JSON) with table titles and page numbers
- Handle multi-page tables and missing titles
- Support for OCR on scanned documents

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd financial-statement-table-detector
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
- Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

## Usage

1. Place your PDF files in the `input` directory
2. Run the script:
```bash
python table_detector.py --input input/ --output output/
```

3. The script will generate:
- A CSV file with table titles and page numbers
- A JSON file with detailed table information

## Configuration

Create a `.env` file in the project root with the following variables:
```
TESSERACT_PATH=/path/to/tesseract
PADDLEOCR_LANG=en
```

## Output Format

### CSV Output
```csv
Table Title,Page Number
"Balance Sheet",1
"Income Statement",2
```

### JSON Output
```json
[
    {
        "title": "Balance Sheet",
        "page_number": 1,
        "confidence": 0.95
    }
]
```

## License

MIT License 