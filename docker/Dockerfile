FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies for Tesseract and other tools
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install sentencepiece for DonutProcessor
RUN pip install sentencepiece

# Pre-download Hugging Face models for CPU
RUN python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten'); VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')"
RUN python -c "from transformers import DonutProcessor, VisionEncoderDecoderModel; DonutProcessor.from_pretrained('naver-clova-ix/donut-base-finetuned-docvqa'); VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base-finetuned-docvqa')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
RUN python -c "from transformers import pipeline; pipeline('text2text-generation', model='google/flan-t5-base', device=-1)"

COPY app/ ./app/
COPY app/ui/streamlit_app.py ./streamlit_app.py

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port 8501"]