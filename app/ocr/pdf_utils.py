from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image
from typing import List
import os

def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    return convert_from_bytes(pdf_bytes)

def convert_pdf_to_images(pdf_path: str, output_folder: str = "/tmp") -> list:
    """
    Convert a PDF file into images, one image per page.

    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Folder to save the output images.

    Returns:
        list: List of file paths to the generated images.
    """
    images = convert_from_path(pdf_path)
    image_paths = []

    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)

    return image_paths

# Example usage
if __name__ == "__main__":
    pdf_path = "/home/intern1/files/sample.pdf"
    image_paths = convert_pdf_to_images(pdf_path)
    print("Generated Images:", image_paths)