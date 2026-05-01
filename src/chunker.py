import json
from pathlib import Path

CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

def estimate_tokens(text):
    return len(text) // 4

def chunk_text(text):
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ".!?" and len(current) > 20:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current_tokens + sentence_tokens > CHUNK_SIZE and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap = []
            overlap_tokens = 0
            for s in reversed(current_chunk):
                if overlap_tokens + estimate_tokens(s) < CHUNK_OVERLAP:
                    overlap.insert(0, s)
                    overlap_tokens += estimate_tokens(s)
                else:
                    break
            current_chunk = overlap
            current_tokens = overlap_tokens
        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if len(c) > 50]

def chunk_documents():
    with open("data/documents.json", encoding="utf-8") as f:
        documents = json.load(f)

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        chunks = chunk_text(doc["content"])
        print(f"  {doc['id']}: {len(chunks)} chunks")

        for i, chunk_content in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"chunk_{chunk_id:04d}",
                "doc_id": doc["id"],
                "doc_title": doc["title"],
                "source": doc["source"],
                "date": doc["date"],
                "chunk_index": i,
                "content": chunk_content,
            })
            chunk_id += 1

    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(all_chunks)} chunks → data/chunks.json")

if __name__ == "__main__":
    chunk_documents()