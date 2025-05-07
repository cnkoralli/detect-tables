# 🧾 Financial Table Title Extractor from PDF Documents

A Python-based solution to process **digital** and **scanned financial statement PDFs**, detect **table regions**, extract their **titles** using OCR and layout analysis, and export the result as a structured CSV file.

---

## 🚀 Features

- ✅ Supports **both scanned and digital** PDFs
- ✅ Detects **table locations** using layout analysis
- ✅ Extracts **titles** using OCR with pattern matching and positional proximity
- ✅ Works across multiple pages
- ✅ Outputs: `Table Title`, `Page Number` as CSV

---

## 📌  Design

```mermaid
flowchart TD
    A[PDF File (Scanned or Digital)] --> B[Convert Pages to Images (pdf2image)]
    B --> C[Run OCR on Each Page (PaddleOCR)]
    A --> D[Run Table Detection (pdfplumber)]
    C & D --> E[For Each Table: Find Text Inside or Around It]
    E --> F[Match Texts Against Title Patterns using Regex]
    F --> G[Export CSV with Table Titles and Page Numbers]
