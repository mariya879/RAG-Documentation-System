import pytesseract
from PIL import Image

def malayalam_ocr(image: Image.Image) -> str:
    return pytesseract.image_to_string(image, lang="mal")