from typing import List
from chromadb import Client
from chromadb.config import Settings

# Initialize ChromaDB client
client = Client(Settings(persist_directory="/home/intern1/files/vector_db"))

# Create or get a collection
collection_name = "document_embeddings"
collection = client.get_or_create_collection(name=collection_name)

def add_embeddings(ids: List[str], embeddings: List[List[float]], metadatas: List[dict]):
    """
    Add embeddings to the vector database.

    Args:
        ids (List[str]): List of unique IDs for the embeddings.
        embeddings (List[List[float]]): List of embedding vectors.
        metadatas (List[dict]): List of metadata dictionaries corresponding to each embedding.
    """
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

def query_embeddings(query_embedding: List[float], n_results: int = 5):
    """
    Query the vector database to find the most similar embeddings.

    Args:
        query_embedding (List[float]): The embedding vector to query.
        n_results (int): Number of similar results to retrieve.

    Returns:
        List[dict]: List of metadata for the most similar embeddings.
    """
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results["metadatas"]

# Example usage
if __name__ == "__main__":
    # Example data
    example_ids = ["doc1", "doc2"]
    example_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    example_metadatas = [{"title": "Document 1"}, {"title": "Document 2"}]

    # Add embeddings
    add_embeddings(example_ids, example_embeddings, example_metadatas)

    # Query embeddings
    query_result = query_embeddings([0.1, 0.2, 0.3])
    print("Query Result:", query_result)
