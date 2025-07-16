from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.ocr.ocr_processor import process_image
from app.embeddings.embedder import generate_embeddings
from app.vectordb.faiss_db import FAISSVectorDB
from app.rag.rag_agent import RAGAgent
import logging

app = FastAPI()

API_KEY = "new_secret_key"

# Initialize FAISS database
dimension = 384  # Assuming embeddings have 384 dimensions
faiss_db = FAISSVectorDB(dimension)

# Initialize RAG agent
rag_agent = RAGAgent()

logging.basicConfig(level=logging.DEBUG)

def check_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

class QueryRequest(BaseModel):
    query: str

@app.post("/process-document/")
def process_document(file: UploadFile = File(...), x_api_key: str = Header(...)):
    logging.debug("Received request to process document.")
    check_api_key(x_api_key)

    # Validate file format
    logging.debug(f"Validating file format for: {file.filename}")
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        logging.error("Unsupported file format.")
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a PNG, JPG, or PDF file.")

    # Save the uploaded file temporarily
    file_path = f"/tmp/{file.filename}"
    logging.debug(f"Saving file to: {file_path}")
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    try:
        # Handle PDF files
        if file.filename.lower().endswith('.pdf'):
            from app.ocr.pdf_utils import convert_pdf_to_images
            logging.debug("Converting PDF to images.")
            image_paths = convert_pdf_to_images(file_path)
            extracted_texts = []
            for image_path in image_paths:
                logging.debug(f"Processing image: {image_path}")
                ocr_result = process_image(image_path)
                extracted_texts.append(ocr_result["extracted_text"])
            extracted_text = "\n".join(extracted_texts)
        else:
            # Process the document with OCR
            logging.debug("Processing image file.")
            ocr_result = process_image(file_path)
            extracted_text = ocr_result["extracted_text"]

        logging.debug(f"Extracted text: {extracted_text}")

        # Generate embeddings for the extracted text
        logging.debug("Generating embeddings.")
        embeddings = generate_embeddings([extracted_text])
        logging.debug(f"Generated embeddings: {embeddings}")

        # Add embeddings to the FAISS database
        logging.debug("Adding embeddings to FAISS database.")
        metadata = {"filename": file.filename, "text": extracted_text}
        faiss_db.add_embeddings(embeddings, [metadata])
        logging.debug(f"FAISS index now contains {faiss_db.index.ntotal} vectors.")

        return {"message": "Document processed successfully", "metadata": metadata}

    except Exception as e:
        logging.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the document.")

@app.post("/query-rag/")
def query_rag(request: QueryRequest, x_api_key: str = Header(...)):
    """
    Endpoint to query the RAG system, retrieve relevant contexts, and generate an answer.

    Args:
        request (QueryRequest): The user query wrapped in a Pydantic model.

    Returns:
        dict: Response containing the generated answer and retrieved contexts.
    """
    check_api_key(x_api_key)

    # Log the query
    print(f"Received query: {request.query}")

    # Retrieve relevant contexts from the FAISS database
    retrieved_contexts = faiss_db.query(request.query, top_k=5)

    # Log retrieved contexts
    print(f"Retrieved contexts: {retrieved_contexts}")

    if not retrieved_contexts:
        raise HTTPException(status_code=404, detail="No relevant contexts found.")

    # Convert numpy.float32 to native Python float in retrieved contexts
    retrieved_contexts = [
        {"metadata": ctx[0], "distance": float(ctx[1])} for ctx in retrieved_contexts
    ]

    # Generate an answer using the RAG agent
    answer = rag_agent.answer(request.query, [ctx["metadata"]["text"] for ctx in retrieved_contexts])

    # Log the generated answer
    print(f"Generated answer: {answer}")

    return {
        "query": request.query,
        "answer": answer,
        "retrieved_contexts": retrieved_contexts
    }