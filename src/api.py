import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json

from vectorstore import VectorStore
from agent import ClimateAgent

app = FastAPI(title="Climate Policy RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading vector store...")
store = VectorStore()
agent = ClimateAgent(store)
print("Ready!")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return FileResponse("outputs/index.html")

@app.get("/documents")
def list_documents():
    with open("data/documents.json", encoding="utf-8") as f:
        return json.load(f)

@app.post("/query")
def query(req: QueryRequest):
    steps = []

    def on_step(event):
        steps.append(event)

    result = agent.run(req.query, on_step=on_step)

    retrieved_docs = []
    web_sources = []
    tool_calls = [s for s in steps if s["type"] == "tool_call"]

    for s in steps:
        if s["type"] == "tool_result":
            if s["tool"] == "retrieve_documents" and isinstance(s["result"], list):
                for r in s["result"]:
                    if isinstance(r, dict) and "doc_title" in r:
                        retrieved_docs.append({
                            "title": r["doc_title"],
                            "source": r.get("source", ""),
                            "score": r.get("score", 0)
                        })
            elif s["tool"] == "web_search" and isinstance(s["result"], list):
                for r in s["result"]:
                    if isinstance(r, dict) and "title" in r:
                        web_sources.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "snippet": r.get("snippet", "")
                        })

    return {
        "answer": result["answer"],
        "iterations": result["iterations"],
        "tokens_used": result["tokens_used"],
        "retrieved_docs": retrieved_docs,
        "web_sources": web_sources,
        "tool_calls": len(tool_calls)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)