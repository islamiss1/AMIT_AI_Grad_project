# 🧠 RAG API — FAISS + SentenceTransformers + Groq (LLaMA 3.3 70B)

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI, FAISS vector search, multilingual embeddings, and Groq-powered LLaMA 70B for high-quality grounded responses.

---

## 🚀 Overview

This system performs semantic document retrieval and uses a large language model to generate context-aware answers.

Instead of letting the LLM hallucinate, we:

1. Convert the user query into embeddings
2. Retrieve relevant document chunks via FAISS
3. Inject those chunks into LLaMA 70B
4. Generate grounded responses

---

## 🏗️ Architecture

```
User Query
     ↓
SentenceTransformer Embedding
     ↓
FAISS Similarity Search
     ↓
Top-K Relevant Chunks
     ↓
Groq LLaMA 3.3 70B
     ↓
Grounded Answer
```

---

## 🧰 Tech Stack

| Layer | Technology |
|--------|------------|
| API | FastAPI |
| Vector Database | FAISS |
| Embeddings | sentence-transformers |
| LLM | Groq (llama-3.3-70b-versatile) |
| Frontend | Static HTML |
| Environment | Python + dotenv |

---

## 📂 Project Structure

```
project-root/
│
├── backend/
│   ├── main.py
│   ├── rag_core/
│   │   ├── llm_client.py
│   │   └── retrieval_ranking.py
│
├── data/
│   └── faiss_db/
│       ├── index.faiss
│       └── chunks.json
│
├── frontend/
│   └── index.html
│
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

Create a `.env` file in your project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/rag-api.git
cd rag-api

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
uvicorn main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

---

## 🔌 API Endpoints

---

### ✅ POST `/rag`

Generate grounded answer.

### Request Body

```json
{
  "query": "What is Islamic finance?",
  "top_k": 4,
  "temperature": 0.7
}
```

### Response

```json
{
  "query": "...",
  "answer": "...",
  "retrieved_chunks": [
    "chunk 1",
    "chunk 2"
  ]
}
```

---

### ✅ GET `/health`

```json
{
  "status": "ok"
}
```

---

## 🧠 Model Configuration

- Embedding Model: `paraphrase-multilingual-MiniLM-L12-v2`
- LLM: `llama-3.3-70b-versatile`
- Default top_k: `4`
- Default temperature: `0.7`

---

## 🐳 Docker Support

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build image:

```bash
docker build -t rag-api .
```

Run container:

```bash
docker run -p 8000:8000 --env-file .env rag-api
```

---

# ☁️ Deployment Guides

---

## 🚀 Deploy on Render

1. Push project to GitHub
2. Create new Web Service on Render
3. Set:
   - Build Command:
     ```
     pip install -r requirements.txt
     ```
   - Start Command:
     ```
     uvicorn main:app --host 0.0.0.0 --port 10000
     ```
4. Add environment variable:
   ```
   GROQ_API_KEY
   ```

---

## 🚀 Deploy on Railway

1. Connect GitHub repo
2. Add environment variable:
   ```
   GROQ_API_KEY
   ```
3. Railway auto-detects Python
4. Set start command:

```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## 🚀 Deploy on EC2 (Ubuntu)

```bash
sudo apt update
sudo apt install python3-pip
pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000
```

(Optional: Use `gunicorn` + `nginx` for production.)

---

# 🔐 Security Notes

- Never expose your GROQ API key
- Use `.env` for secrets
- In production, use HTTPS
- Consider rate limiting

---

# 📈 Future Improvements

- Add streaming responses
- Add conversation memory
- Add hybrid search (BM25 + dense)
- Add reranker model
- Add authentication (JWT)
- Add caching layer (Redis)
- Add logging & monitoring

---

# 🎯 Use Cases

- Knowledge base chatbot
- Domain-specific assistant
- Islamic finance Q&A
- Legal document search
- Internal enterprise assistant
- Multilingual search system

---

# 👨‍💻 Author

Built with ❤️ using FastAPI + FAISS + Groq.

---

# 📜 License

Add your license here (MIT recommended).

---

