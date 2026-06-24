from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from rag import query_documents, index_documents
import os

app = FastAPI(title="SORIB API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class Question(BaseModel):
    question: str
    history: List[Message] = []

@app.on_event("startup")
async def startup():
    print("Indexando documentos...")
    index_documents()
    print("Documentos indexados correctamente.")

@app.post("/ask")
async def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    history = [{"role": m.role, "content": m.content} for m in q.history]
    result = query_documents(q.question, history)
    return {"answer": result["answer"], "sources": result["sources"]}

@app.get("/health")
async def health():
    return {"status": "ok", "message": "SORIB está funcionando"}