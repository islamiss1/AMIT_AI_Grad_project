import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_core.llm_client import GroqLLM
from rag_core.retireval_ranking import SemanticRetriever
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

# ----------------------------
# App initialization
# ----------------------------
app = FastAPI(
    title="RAG API",
    description="FAISS + SentenceTransformers + Groq (LLaMA 70B)",
    version="1.0.0"
)


app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# ----------------------------
# Config (same as main.py)
# ----------------------------
FAISS_INDEX_PATH = r"D:\project islam\data\faiss_db\index.faiss"
CHUNKS_PATH = r"D:\project islam\data\faiss_db\chunks.json"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY environment variable not set")

# ----------------------------
# Initialize RAG components ONCE
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def serve_chat():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()


retriever = SemanticRetriever(
    faiss_index_path=FAISS_INDEX_PATH,
    chunks_path=CHUNKS_PATH,
    embedding_model=EMBEDDING_MODEL
)

llm = GroqLLM(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL
)

# ----------------------------
# Request / Response models
# ----------------------------
class RAGRequest(BaseModel):
    query: str
    top_k: int = 4
    temperature: float = 0.7  


class RAGResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: list[str]


# ----------------------------
# RAG Endpoint
# ----------------------------
@app.post("/rag", response_model=RAGResponse)
def generate_rag_answer(request: RAGRequest):
    try:
        chunks = retriever.retrieve(
            query=request.query,
            top_k=request.top_k
        )

        answer = llm.generate(
            query=request.query,
            context_chunks=chunks,
            temperature=request.temperature  
        )

        return RAGResponse(
            query=request.query,
            answer=answer,
            retrieved_chunks=chunks
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Health check
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

