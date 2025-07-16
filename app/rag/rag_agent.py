from transformers import pipeline
import logging
from typing import List

logging.basicConfig(level=logging.DEBUG)

class RAGAgent:
    def __init__(self, model_name="google/flan-t5-base"):
        self.generator = pipeline("text2text-generation", model=model_name, device=-1)
        self.max_length = 512  # Maximum sequence length supported by the model

    def truncate_context(self, context: str) -> str:
        """
        Truncate the context to fit within the model's maximum sequence length.

        Args:
            context (str): The input context string.

        Returns:
            str: The truncated context.
        """
        tokens = context.split()
        if len(tokens) > self.max_length:
            return " ".join(tokens[:self.max_length])
        return context

    def answer(self, query: str, retrieved_contexts: List[str]) -> str:
        context = "\n".join(retrieved_contexts)
        truncated_context = self.truncate_context(context)
        logging.debug(f"Truncated context: {truncated_context}")
        prompt = f"Context: {truncated_context}\n\nQuestion: {query}\nAnswer:"
        logging.debug(f"Generated prompt: {prompt}")
        result = self.generator(prompt, max_new_tokens=256)
        logging.debug(f"Generation result: {result}")
        return result[0]["generated_text"]

if __name__ == "__main__":
    agent = RAGAgent()
    query = "What is the capital of France?"
    retrieved_contexts = [
        "France is a country in Europe.",
        "Paris is the capital city of France."
    ]
    answer = agent.answer(query, retrieved_contexts)
    print(f"Query: {query}")
    print(f"Answer: {answer}")