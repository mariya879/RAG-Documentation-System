from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

# Use Donut model for visual extraction (charts, tables)
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

def extract_visual_text(image: Image.Image) -> str:
    pixel_values = processor(image, return_tensors="pt").pixel_values
    task_prompt = "<s_docvqa>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return result