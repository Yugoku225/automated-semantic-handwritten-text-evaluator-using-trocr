import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

def clean_text_basic(text):
    """
    Cleans OCR text using regex. Removes special characters, extra newlines, 
    and normalizes spacing.
    """
    # Replace newlines and tabs with spaces
    text = re.sub(r'[\n\t]+', ' ', text)
    # Remove non-alphanumeric characters except basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', text)
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_and_remove_stopwords(text):
    """
    Tokenizes text and removes common English stopwords.
    Returns a list of clean tokens.
    """
    # Lowercase the text
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and pure punctuation tokens
    stop_words = set(stopwords.words('english'))
    clean_tokens = [
        word for word in tokens 
        if word not in stop_words and word.isalnum()
    ]
    
    return clean_tokens

def correct_spelling(tokens):
    """
    Optional: attempts to correct spelling of a list of tokens.
    Note: Can be slow for large texts and might alter domain-specific terms.
    """
    spell = SpellChecker()
    corrected_tokens = []
    
    for word in tokens:
        # Get the most likely correction
        corrected = spell.correction(word)
        if corrected is not None:
             corrected_tokens.append(corrected)
        else:
             corrected_tokens.append(word)
             
    return corrected_tokens

def process_student_answer(raw_ocr_text):
    """
    Runs the full Phase 2 NLP pipeline on raw OCR text.
    Returns a dictionary with raw, cleaned, and tokenized versions.
    """
    print("[*] Running Phase 2 NLP Pipeline...")
    
    # 1. Regex Basic Cleaning
    cleaned_string = clean_text_basic(raw_ocr_text)
    
    # 2. Tokenization and Stopword Removal
    tokens = tokenize_and_remove_stopwords(cleaned_string)
    
    # 3. Spelling Correction (Optional, turned on for demonstration)
    print("    -> Correcting spelling...")
    final_tokens = correct_spelling(tokens)
    
    # Reconstruct the string from tokens for later Semantic Similarity phase
    processed_string = " ".join(final_tokens)
    
    return {
        "raw_text": raw_ocr_text,
        "regex_cleaned_text": cleaned_string,
        "tokens": final_tokens,
        "final_processed_string": processed_string
    }
