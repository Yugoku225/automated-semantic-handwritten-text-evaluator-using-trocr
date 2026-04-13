import cv2
import pytesseract
import numpy as np
import os

def preprocess_image(image_path, output_dir="data/processed_images"):
    """
    Reads an image and applies standard preprocessing to improve OCR accuracy.
    Includes grayscale conversion, blur (to remove noise), and thresholding.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find image at {image_path}")

    # 1. Read image
    img = cv2.imread(image_path)
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply Gaussian blur to smooth out noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Apply Otsu's thresholding for binarization (makes background white, text black)
    # Using adaptive thresholding can also work well for varying lighting.
    # Here we use Adaptive Threshold for handwritten text which is often better than global Otsu.
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Save processed image for debugging
    filename = os.path.basename(image_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    processed_path = os.path.join(output_dir, f"processed_{filename}")
    cv2.imwrite(processed_path, thresh)
    
    return thresh

def extract_text_from_image(preprocessed_image):
    """
    Uses Tesseract OCR to extract text from a processed OpenCV image.
    """
    # psm 6 assumes a single uniform block of text.
    # psm 3 or 4 could also be used depending on image structure.
    custom_config = r'--oem 3 --psm 6'
    
    extracted_text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    return extracted_text.strip()

def run_ocr_pipeline(image_path):
    """
    Full pipeline: Preprocess -> Extract Text -> Return Text
    """
    print(f"[*] Processing image: {image_path}")
    processed_img = preprocess_image(image_path)
    
    print("[*] Extracting text using Tesseract...")
    text = extract_text_from_image(processed_img)
    
    return text
