import PyPDF2
import os

def extract_text_from_pdf(pdf_path):
    """
    Extracts all textual content from a given PDF file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF missing: {pdf_path}")
        
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        print(f"[!] Error reading PDF: {e}")
        return ""
        
    return text.strip()
