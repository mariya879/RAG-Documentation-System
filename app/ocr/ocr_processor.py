import pytesseract
from pytesseract import Output
from PIL import Image
import cv2
import numpy as np
from spellchecker import SpellChecker

def process_image(image_path: str, lang: str = "eng+mal") -> dict:
    """
    Process an image to extract text using Tesseract OCR.

    Args:
        image_path (str): Path to the image file.
        lang (str): Language(s) for OCR (default: English and Malayalam).

    Returns:
        dict: Extracted text and bounding box information.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}. Ensure the file exists and is a valid image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocess the image (e.g., thresholding)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Apply noise removal
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # Apply adaptive thresholding for better handwritten text recognition
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Deskew the image
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Adjust contrast
    contrast_enhanced = cv2.convertScaleAbs(deskewed, alpha=1.5, beta=0)

    # Configure Tesseract to use a custom dictionary for Indian names
    custom_config = "--user-words /home/intern1/files/app/ocr/indian_names.txt"

    # Perform OCR with enhanced preprocessing
    data = pytesseract.image_to_data(contrast_enhanced, lang=lang, config=custom_config, output_type=Output.DICT)

    # Extract text and bounding boxes
    extracted_text = " ".join(data['text']).strip()
    bounding_boxes = [
        {
            "text": data['text'][i],
            "x": data['left'][i],
            "y": data['top'][i],
            "width": data['width'][i],
            "height": data['height'][i]
        }
        for i in range(len(data['text'])) if data['text'][i].strip()
    ]

    # Post-processing: Spell-checking and custom rules
    spell = SpellChecker()

    # Correct spelling errors in extracted text
    corrected_text = " ".join([
        spell.correction(word) if word.isalpha() and spell.correction(word) is not None else word
        for word in extracted_text.split()
    ])

    # Apply custom rules for common OCR misinterpretations
    corrected_text = corrected_text.replace("\\CHELOR", "BACHELOR")
    corrected_text = corrected_text.replace("Jniversity", "University")

    # Update the extracted text with corrections
    extracted_text = corrected_text

    return {
        "extracted_text": extracted_text,
        "bounding_boxes": bounding_boxes
    }

# Example usage
if __name__ == "__main__":
    image_path = "/home/intern1/files/image2.png"
    result = process_image(image_path)
    print("Extracted Text:", result["extracted_text"])
    print("Bounding Boxes:", result["bounding_boxes"])
