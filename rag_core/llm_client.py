import os
import json
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from groq import Groq




# ----------------------------
# LLM Client (Groq)
# ----------------------------
class GroqLLM:
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, query: str, context_chunks: List[str], temperature: float = 0.7) -> str:

        context = "\n\n---\n\n".join(context_chunks)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Use the context below to answer the question. "
                    "You may also use your general knowledge if needed."
                )  

            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{query}
"""
            }
        ]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature  # ← Uses the value from frontend

        )

        return completion.choices[0].message.content.strip()


