import pytesseract
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# For simple English handwritten OCR fallback
def tesseract_ocr(image: Image.Image) -> str:
    return pytesseract.image_to_string(image, lang="eng")

# For better results, use TrOCR (if GPU RAM allows)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def trocr_ocr(image: Image.Image) -> str:
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]