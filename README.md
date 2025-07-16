# RAG Pipeline for Document Processing

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline to process scanned multi-page documents containing:
- Handwritten text (English and Malayalam)
- Visual elements such as charts and graphs

The system extracts and structures content, generates embeddings, stores them in a vector database, and provides a chat interface for querying the processed documents.

## Features
- **Document Processing**:
  - OCR for handwritten English and Malayalam text
  - Extraction of text from charts and graphs using the Donut model
- **Vector Database**:
  - Embedding generation using Hugging Face models
  - Storage and retrieval using FAISS
- **RAG System**:
  - Context retrieval and answer generation using a FastAPI-based agent
- **Chat Interface**:
  - Streamlit-based UI for document upload and querying
- **Containerization**:
  - Fully containerized using Docker

## Tech Stack
- **Programming Language**: Python
- **Frameworks**: FastAPI, Streamlit
- **Libraries**: Hugging Face Transformers, FAISS, Tesseract OCR
- **Containerization**: Docker

## Setup Instructions

### Prerequisites
- Docker installed on your system

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Build the Docker image:
   ```bash
   docker build -t rag-pipeline -f docker/Dockerfile .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 8000:8000 -p 8501:8501 rag-pipeline
   ```

4. Access the application:
   - FastAPI API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Streamlit UI: [http://127.0.0.1:8501](http://127.0.0.1:8501)

## Testing
The system has been tested with:
- Handwritten English text, including Indian names
- Handwritten Malayalam text
- Charts and graphs

## Project Structure
```
.
├── app
│   ├── api
│   │   └── main.py          # FastAPI endpoints
│   ├── charts
│   │   └── chart_extractor.py # Chart/graph text extraction
│   ├── embeddings
│   │   └── embedder.py      # Embedding generation
│   ├── ocr
│   │   ├── ocr_processor.py # OCR logic
│   │   └── pdf_utils.py     # PDF-to-image conversion
│   ├── rag
│   │   └── rag_agent.py     # RAG agent for answer generation
│   ├── ui
│   │   └── streamlit_app.py # Streamlit chat interface
│   └── vectordb
│       └── faiss_db.py      # FAISS vector database logic
├── docker
│   └── Dockerfile           # Docker configuration
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Contact
For any questions or issues, please contact Mariya Johnson at mariyajohnson879@gmail.com
