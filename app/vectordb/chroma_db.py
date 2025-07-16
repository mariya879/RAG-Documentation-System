import chromadb
from chromadb.utils import embedding_functions

class ChromaVectorDB:
    def __init__(self, collection_name="documents"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
        )

    def add_document(self, doc_id, text, metadata=None):
        self.collection.add(documents=[text], ids=[doc_id], metadatas=[metadata or {}])

    def query(self, query_text, n_results=3):
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        return results["documents"][0], results["ids"][0]