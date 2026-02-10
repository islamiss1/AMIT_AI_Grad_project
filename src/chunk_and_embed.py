import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os


# ----------------------------
# Load cleaned JSON
# ----------------------------
def load_json(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Chunking: 1 object = 1 chunk
# ----------------------------
def create_chunks(data: List[Dict]) -> List[str]:
    chunks = []

    for item in data:
        chunk = (
            f"Question: {item['question']}\n"
            f"Long Answer: {item['long_answers']}\n"
            f"Short Answer: {item['short_answers']}"
        )
        chunks.append(chunk)

    return chunks


# ----------------------------
# Create FAISS vector DB
# ----------------------------
def build_faiss_index(
    chunks: List[str],
    save_dir: str = "faiss_db",
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
):
    os.makedirs(save_dir, exist_ok=True)

    # Load embedding model
    model = SentenceTransformer(model_name)

    # Create embeddings
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # VERY important for cosine similarity
    )

    dim = embeddings.shape[1]

    # FAISS index (cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, os.path.join(save_dir, "index.faiss"))

    # Save chunks (metadata)
    with open(os.path.join(save_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ FAISS DB saved to '{save_dir}'")
    print(f"📊 Total chunks indexed: {len(chunks)}")


# ----------------------------
# Run pipeline
# ----------------------------
if __name__ == "__main__":
    DATA_PATH = r"D:\project islam\data\preprocessed\cleaned_data.json"  # cleaned JSON from previous step

    data = load_json(DATA_PATH)
    chunks = create_chunks(data)

    build_faiss_index(chunks)
