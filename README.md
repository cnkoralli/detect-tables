### **Title: Table Title Detection from Scanned and Digital Financial PDFs**

---

### **1. High-Level Design**

This project automatically detects **tables** and extracts their **titles** from financial PDF documents (both scanned and digital). It uses layout analysis, OCR, and text pattern recognition to identify key financial statements.

#### **Processing Flow:**

```
            +-------------------+
            |    Input PDF      |
            +--------+----------+
                     |
                     v
        +------------+------------+
        | Determine if PDF is     |
        | scanned or text-based   |
        +------------+------------+
                     |
          +----------+----------+
          |                     |
          v                     v
  [Scanned PDF]          [Digital PDF]
  Image conversion       Text extraction (PyMuPDF, pdfplumber)
  + table detection      + regex pattern search
  + OCR on titles        + title matching
                     \         /
                      \       /
                     +---------+
                     | Filtered |
                     |  Titles  |
                     +----+----+
                          |
                          v
                 Output CSV/JSON File
```

---

### **2. Implementation Details**

#### **Technologies Used:**

| Tool/Library     | Purpose                                                            |
| ---------------- | ------------------------------------------------------------------ |
| `pdf2image`      | Converts scanned PDFs to high-resolution images                    |
| `pytesseract`    | OCR engine for extracting text from images                         |
| `opencv-python`  | Detects table regions using morphological operations               |
| `PyMuPDF (fitz)` | Parses and extracts text from digital PDFs                         |
| `pdfplumber`     | Alternative method to extract structured tables from digital PDFs  |
| `re` (regex)     | Matches known table title patterns like “Balance Sheet”, “Profit…” |
| `pandas`         | Organizes final output and writes to CSV                           |
| `argparse`       | CLI interface to pass input/output file paths                      |
| `logging`        | Provides structured feedback and debugging info                    |

#### **Why These Tools?**

* Tesseract + OpenCV are powerful for image-based OCR + layout detection.
* PyMuPDF and pdfplumber offer fast, accurate text-level access for digital PDFs.
* Regex rules allow for flexible and customizable title recognition.

---

### **3. How to Build and Run the Project**

#### ✅ **Step 1: Install Required Packages**

Use the terminal inside Replit or your local environment:

```bash
pip install pdf2image pytesseract opencv-python pillow fitz pdfplumber pandas
```

Also install Tesseract OCR on your system:

* Windows: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
* macOS: `brew install tesseract`
* Linux: `sudo apt install tesseract-ocr`

Install **poppler** (for `pdf2image`):

* Windows: [https://github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases)
* macOS: `brew install poppler`
* Linux: `sudo apt install poppler-utils`

---

#### ✅ **Step 2: Prepare Your Project Folder**

Structure:

```
project/
├── main.py
├── sample.pdf
├── output.csv
└── README.md
```

---

#### ✅ **Step 3: Run the Script**

```bash
python main.py sample.pdf --output output.csv
```

This will:

* Detect if the PDF is scanned or digital.
* Apply appropriate extraction logic.
* Save the output titles with page numbers in a clean CSV.

---

### **4. Sample Output**

```csv
Table Title,Page Number
Balance Sheet as at 31st March 2023,2
Statement of Profit and Loss for the year ended 31st March 2023,4
Notes to Financial Statements,6
```

---

### **5. Customization & Extensions**

* Add support for more title patterns in `TABLE_PATTERNS`.
* Export results to JSON by modifying `extract_table_titles`.
* Add image export of cropped table areas.
* Wrap into a web UI using Streamlit.

---

