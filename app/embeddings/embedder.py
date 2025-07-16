from sentence_transformers import SentenceTransformer
from typing import List

# Multilingual model supports Malayalam
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_embedding(text: str):
    return model.encode([text])[0]

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using the pre-loaded Sentence Transformer model.

    Args:
        texts (List[str]): List of text strings to embed.

    Returns:
        List[List[float]]: List of embeddings for the input texts.
    """
    return model.encode(texts, convert_to_numpy=True)

# Example usage
if __name__ == "__main__":
    sample_texts = [
        "France is a country in Europe.",
        "Paris is the capital city of France."
    ]
    embeddings = generate_embeddings(sample_texts)
    print("Generated Embeddings:", embeddings)