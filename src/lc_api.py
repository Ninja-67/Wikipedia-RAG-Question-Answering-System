from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import os
from .lc_chain import ask as lc_ask

VS_DIR = os.getenv("VS_DIR", "artifacts/lc_faiss")
PROVIDER = os.getenv("PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

app = FastAPI(title="Wikipedia RAG API (LangChain-only)")

class AskRequest(BaseModel):
    question: str
    k: int = 8
    rerank_top: int = 20

class AskResponse(BaseModel):
    answer: str
    contexts: List[Dict]

@app.get("/health")
def health():
    return {"status": "ok", "backend": "langchain"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    out = lc_ask(VS_DIR, req.question, k=req.k, rerank_top=req.rerank_top,
                 provider=PROVIDER, openai_model=OPENAI_MODEL, hf_model=HF_MODEL,
                 max_tokens=MAX_TOKENS, temperature=TEMPERATURE, embed_model=EMBED_MODEL)
    return AskResponse(**out)
