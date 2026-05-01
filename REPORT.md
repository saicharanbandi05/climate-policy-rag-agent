# Climate Policy RAG Agent — Evaluation & Reflection Report

**Author:** Hema Sai Charan Bandi  
**Project:** Agent-Based Policy Question Answering System  
**Topic:** Climate Change Policy  

---

## What I Built

A web-based AI agent that answers questions about climate change policy.
The agent searches a curated database of 10 real policy documents and
can also search the web for recent news and events.

You can ask it questions like:
- "What is the Paris Agreement temperature target?"
- "How much does the IRA invest in clean energy?"
- "What are the latest climate news headlines today?"

---

## How It Works — Step by Step

User types a question
↓
Step 1: Agent searches 10 policy documents (always)
↓
Step 2: If question is about recent news → also searches the web
↓
Step 3: LLM reads both results and writes a cited answer
↓
Answer appears in the web UI with sources shown
---

## Tech Stack

| Component | Tool Used | Why |
|-----------|-----------|-----|
| LLM (brain) | Groq LLaMA 3 70B | Free, fast, powerful |
| Embeddings | sentence-transformers | Runs locally, no API cost |
| Vector Database | ChromaDB | Stores document embeddings |
| Web Search | DuckDuckGo | Free, no API key needed |
| Backend API | FastAPI | Fast Python web server |
| Frontend | HTML + CSS + JavaScript | Simple, no framework needed |

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

## Test Results

I tested the agent with 7 questions. Here are the results:

| Question | Source | Result |
|----------|--------|--------|
| Paris Agreement temperature target? | Corpus | ✅ Correct — 1.5°C and 2°C cited |
| How much does the IRA invest? | Corpus | ✅ Correct — $369 billion cited |
| What is the Justice40 Initiative? | Corpus | ✅ Correct — 40% benefit target cited |
| How potent is methane vs CO2? | Corpus | ✅ Correct — 80x over 20 years cited |
| Latest climate news today? | Web Search | ✅ Current web results returned |
| COP28 Loss and Damage Fund? | Corpus | ✅ Correct — $700 million cited |
| US healthcare reform policy? | Neither | ✅ Correctly said it doesn't know |

**7 out of 7 questions answered correctly.**

---

## Strengths

### 1. Finds Meaning, Not Just Keywords
Traditional search looks for exact word matches. Our system uses
embeddings — it understands meaning. So "warming goal" correctly
finds documents about "temperature targets" even though the words
are different.

### 2. Web Search for Current Events
The corpus was written in 2022-2023. For questions about recent
news, the agent automatically searches the web using DuckDuckGo
and combines those results with the corpus.

### 3. Always Cites Sources
Every answer includes which document or website the information
came from. This makes answers trustworthy and verifiable.

### 4. Completely Free to Run
- Groq API — free tier
- DuckDuckGo search — free, no API key
- Embedding model — runs on your computer, no cost

### 5. Clean Modular Code
Each file has one job:
- `chunker.py` — splits documents
- `vectorstore.py` — handles embeddings and search
- `agent.py` — runs the retrieval logic
- `api.py` — serves the web API
- `index.html` — the user interface

---

## Limitations

### 1. Small Corpus — Only 10 Documents
The agent only knows what's in these 10 documents. Deep or
specific questions about regional policies, specific legislation
clauses, or newer policies will not be answered well.

### 2. Web Search is Keyword-Triggered
Web search only activates when the question contains words like
"latest", "today", "recent", or "current". If someone asks
"Who won the climate vote last week?" without those trigger
words, it won't search the web.

### 3. No Memory Between Questions
Each question is answered independently. If you ask "Tell me
more about that", the agent doesn't remember what "that" refers
to from the previous answer.

### 4. One Chunk Per Document Originally
The first version of chunking created only 1 chunk per document.
This was improved to create 3-5 chunks per document for better
retrieval precision.

### 5. No Answer Quality Score
There is no automated way to score whether an answer is correct.
Testing was done manually by checking if expected facts appeared
in the response.

---

## How to Make It Better — Production Improvements

| Problem | Current Approach | Production Solution |
|---------|-----------------|---------------------|
| Only 10 documents | Hand-written JSON | Auto-scrape 100+ real policy URLs |
| Simple chunking | Fixed sentence splitting | Smart chunking that keeps ideas together |
| Basic retrieval | Single ChromaDB search | Search + re-ranking for better results |
| Keyword web trigger | Simple if/else check | Let the LLM decide when to search web |
| No conversation memory | Stateless per query | Save chat history with Redis |
| Manual evaluation | Read answers yourself | Automated scoring with another LLM |
| Local only | Runs on your laptop | Deploy to cloud with Docker |
| No monitoring | No logs or metrics | Add dashboards for usage and errors |
| No user accounts | Anyone can use it | Add login and usage limits |

---

## Conclusion

This project successfully demonstrates an agentic RAG system that:

1. Retrieves relevant policy information using semantic search
2. Supplements with live web search when needed
3. Synthesizes cited answers using a powerful LLM
4. Presents results in a clean web interface

The system correctly answered all 7 test questions and handles
out-of-corpus questions gracefully — either by searching the web
or clearly stating it doesn't have the information.

The modular design makes it straightforward to expand with more
documents, better retrieval, and production infrastructure when needed.