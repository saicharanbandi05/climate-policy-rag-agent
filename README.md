# 🌍 Climate Policy RAG Agent

An agent-based Retrieval-Augmented Generation (RAG) system that answers
questions about climate change policy using a curated corpus of 10
authoritative documents — with live web search for current events.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)

---

## What It Does

- Ask any question about climate policy and get a cited answer
- Searches 10 real policy documents using semantic search
- Automatically searches the web for recent news and current events
- Shows which sources were used for every answer

---

## Demo

> **Question:** What is the Paris Agreement temperature target?

> **Answer:** The Paris Agreement aims to limit global temperature increases
> to well below 2°C above pre-industrial levels, pursuing efforts to limit
> warming to 1.5°C. *(Source: Paris Agreement – Key Provisions, UNFCCC 2015)*

> **Question:** What are the latest climate news headlines today?

> **Answer:** *(searches web in real time and returns current news)*

---

## Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Groq LLaMA 3 70B (free) |
| Embeddings | sentence-transformers (local) |
| Vector Database | ChromaDB |
| Web Search | DuckDuckGo (free, no key) |
| Backend | FastAPI |
| Frontend | HTML + CSS + JavaScript |

---

## The 10 Policy Documents

1. Paris Agreement – Key Provisions (2015)
2. Inflation Reduction Act – Climate Provisions (2022)
3. IPCC Sixth Assessment Report (2021-2022)
4. European Union Green Deal (2019-2023)
5. Carbon Pricing Mechanisms – Global Status (2023)
6. Environmental Justice and Climate Policy (2023)
7. Renewable Energy Transition – Costs and Jobs (2023)
8. U.S. National Climate Assessment (2023)
9. Global Methane Pledge (2021-2023)
10. Climate Finance and Developing Nations (2023)

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/saicharanbandi05/climate-policy-rag-agent.git
cd climate-policy-rag-agent
```

### 2. Create virtual environment
```bash
python -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a `.env` file in the root folder:
GROQ_API_KEY=your_groq_key_here

Get a free key at: https://console.groq.com

### 5. Build the vector store
```bash
python src/chunker.py
python src/vectorstore.py
```

### 6. Run the app
```bash
python src/api.py
```

Open your browser at: **http://localhost:8000**

---

## Project Structure
climate-policy-rag-agent/
├── src/
│   ├── agent.py          # Core agent logic
│   ├── vectorstore.py    # ChromaDB embeddings
│   ├── chunker.py        # Document chunking
│   ├── cli.py            # Terminal interface
│   └── api.py            # FastAPI backend
├── data/
│   ├── documents.json    # 10 policy documents
│   └── chunks.json       # Chunked documents
├── outputs/
│   └── index.html        # Web UI
├── REPORT.md             # Evaluation and reflection
├── requirements.txt      # Dependencies
└── .env.example          # API key template
---

## How It Works
User Question
↓
Stage 1: Semantic search over ChromaDB corpus (always)
↓
Stage 2: DuckDuckGo web search (if query mentions recent/current events)
↓
Stage 3: Groq LLaMA 3 synthesizes answer from both sources
↓
Cited answer shown in web UI

---

## Author

**Hema Sai Charan Bandi**
- GitHub: [@saicharanbandi05](https://github.com/saicharanbandi05)
