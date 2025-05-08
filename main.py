import os
import detector
from typing import List, Dict, Any

def validate_pdf(pdf_path: str) -> bool:
    """
    Validate if the file exists and is a PDF
    """
    if not os.path.exists(pdf_path):
        print("Error: File does not exist!")
        return False
        
    if not pdf_path.lower().endswith('.pdf'):
        print("Error: File must be a PDF!")
        return False
        
    return True

def main():
    try:
        # Get PDF path from user
        pdf_path = input("Enter the path to your PDF file: ")
        
        # Validate PDF
        if not validate_pdf(pdf_path):
            return
        
        # Process PDF
        print("\nProcessing PDF...")
        try:
            detector.process_pdf_with_gemini(pdf_path)
            print("Tables extracted successfully!")
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return
            
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main() 