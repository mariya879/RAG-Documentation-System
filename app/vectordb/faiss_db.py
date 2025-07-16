import faiss
import numpy as np
from typing import List, Tuple
from app.embeddings.embedder import generate_embeddings

class FAISSVectorDB:
    def __init__(self, dimension: int):
        """
        Initialize a FAISS vector database.

        Args:
            dimension (int): The dimensionality of the vectors.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []  # To store metadata corresponding to vectors

    def add_embeddings(self, embeddings: List[List[float]], metadatas: List[dict]):
        """
        Add embeddings and their metadata to the FAISS index.

        Args:
            embeddings (List[List[float]]): List of embedding vectors.
            metadatas (List[dict]): List of metadata dictionaries corresponding to each embedding.
        """
        print(f"Adding {len(embeddings)} embeddings to FAISS index.")
        print(f"Embeddings shape: {np.array(embeddings).shape}")
        vectors = np.array(embeddings).astype('float32')
        self.index.add(vectors)
        print(f"FAISS index now contains {self.index.ntotal} vectors.")
        self.metadata.extend(metadatas)
        print(f"Metadata size: {len(self.metadata)}")

    def query_embeddings(self, query_embedding: List[float], n_results: int = 5) -> List[Tuple[dict, float]]:
        """
        Query the FAISS index to find the most similar embeddings.

        Args:
            query_embedding (List[float]): The embedding vector to query.
            n_results (int): Number of similar results to retrieve.

        Returns:
            List[Tuple[dict, float]]: List of tuples containing metadata and distances for the most similar embeddings.
        """
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, n_results)

        # Log indices and distances for debugging
        print("Indices:", indices)
        print("Distances:", distances)

        # Ensure indices and distances are valid
        if len(indices) == 0 or len(indices[0]) == 0:
            return []
        if len(distances) == 0 or len(distances[0]) == 0:
            return []

        # Filter results to ensure valid metadata indices
        results = [
            (self.metadata[i], distances[0][j])
            for j, i in enumerate(indices[0])
            if i < len(self.metadata)
        ]
        return results

    def query(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Query the FAISS index to retrieve the most relevant contexts for a given query string.

        Args:
            query (str): The query string.
            top_k (int): Number of top results to retrieve.

        Returns:
            List[dict]: List of metadata dictionaries for the most similar embeddings.
        """
        # Check if the FAISS index is empty
        if self.index.ntotal == 0:
            print("FAISS index is empty. No embeddings to query.")
            return []

        # Log the number of vectors in the FAISS index
        print(f"FAISS index contains {self.index.ntotal} vectors.")

        # Log metadata for debugging
        print("Stored metadata:", self.metadata)

        # Generate embedding for the query string
        query_embedding = generate_embeddings([query])[0]

        # Log the query embedding
        print("Query embedding:", query_embedding)

        # Retrieve the most similar embeddings
        results = self.query_embeddings(query_embedding, n_results=top_k)

        # Filter out invalid indices (-1)
        valid_results = [result for result in results if result[0] is not None]

        return valid_results

# Example usage
if __name__ == "__main__":
    # Initialize FAISS database
    dimension = 128
    faiss_db = FAISSVectorDB(dimension)

    # Example data
    example_embeddings = np.random.random((10, dimension)).tolist()
    example_metadatas = [{"id": f"doc{i}"} for i in range(10)]

    # Add embeddings
    faiss_db.add_embeddings(example_embeddings, example_metadatas)

    # Query embeddings
    query_result = faiss_db.query_embeddings(example_embeddings[0], n_results=3)
    print("Query Result:", query_result)
