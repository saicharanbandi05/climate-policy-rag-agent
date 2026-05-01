import json
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "climate_policy"
MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "data/chroma_db"

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks):
        existing_ids = set(self.collection.get()["ids"])
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        if not new_chunks:
            print("All chunks already in vector store.")
            return
        print(f"Embedding {len(new_chunks)} chunks...")
        for i in range(0, len(new_chunks), 50):
            batch = new_chunks[i:i+50]
            self.collection.add(
                ids=[c["chunk_id"] for c in batch],
                documents=[c["content"] for c in batch],
                metadatas=[{
                    "doc_id": c["doc_id"],
                    "doc_title": c["doc_title"],
                    "source": c["source"],
                    "date": c["date"],
                    "chunk_index": c["chunk_index"],
                } for c in batch],
            )
            print(f"  Embedded {min(i+50, len(new_chunks))}/{len(new_chunks)}")
        print(f"Done! {self.collection.count()} chunks in store.")

    def retrieve(self, query, top_k=4):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "score": round(1 - results["distances"][0][i], 4),
                **results["metadatas"][0][i],
            })
        return hits

    def count(self):
        return self.collection.count()

def build_vectorstore():
    with open("data/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    store = VectorStore()
    store.add_chunks(chunks)
    return store

if __name__ == "__main__":
    store = build_vectorstore()
    print("\n--- Test Retrieval ---")
    results = store.retrieve("Paris Agreement temperature target", top_k=3)
    for r in results:
        print(f"  [{r['score']}] {r['doc_title']}")
        print(f"  {r['content'][:100]}...")