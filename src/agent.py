"""
agent.py
--------
Climate Policy RAG Agent

Uses a two-stage retrieval approach:
  1. Always searches the internal corpus (ChromaDB + embeddings)
  2. Optionally searches the web for recent/current information

Author: Hema Sai Charan Bandi
"""

import json
import os

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from groq import Groq

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

MODEL = "llama-3.3-70b-versatile"
TOP_K = 5  # number of corpus chunks to retrieve per query

# Keywords that trigger a web search in addition to corpus retrieval
WEB_TRIGGER_KEYWORDS = [
    "latest", "today", "recent", "current",
    "news", "summit", "2025", "2026"
]

SYSTEM_PROMPT = """You are a climate policy analyst assistant with two knowledge sources:

1. INTERNAL CORPUS — 10 authoritative climate policy documents covering:
   Paris Agreement, Inflation Reduction Act, IPCC AR6, EU Green Deal,
   Carbon Pricing, Environmental Justice, Renewable Energy, US National
   Climate Assessment, Methane Policy, and Climate Finance.

2. WEB SEARCH — for current events, recent news, or topics not in the corpus.

## Rules
- Prioritize internal corpus documents when answering policy questions.
- Use web results to supplement with recent or missing information.
- Always cite your sources (document title or website name).
- Be precise — include specific numbers, dates, and percentages.
- If the answer is not in either source, say so clearly.
"""


# ── Agent ────────────────────────────────────────────────────────────────────

class ClimateAgent:
    """
    Two-stage retrieval agent for climate policy Q&A.

    Stage 1: Semantic search over internal ChromaDB corpus.
    Stage 2: Optional DuckDuckGo web search for recent/current queries.
    Both results are passed to the LLM for final answer synthesis.
    """

    def __init__(self, vector_store):
        """
        Args:
            vector_store: VectorStore instance with embedded policy documents.
        """
        self.store = vector_store
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _needs_web_search(self, query: str) -> bool:
        """Return True if the query contains keywords suggesting recent info."""
        return any(kw in query.lower() for kw in WEB_TRIGGER_KEYWORDS)

    def _web_search(self, query: str) -> list:
        """
        Search the web using DuckDuckGo.

        Args:
            query: Search query string.

        Returns:
            List of dicts with keys: title, snippet, url.
        """
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=4))
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                }
                for r in raw
            ]
        except Exception as e:
            return [{"error": f"Web search failed: {str(e)}"}]

    def _build_context(self, query: str, corpus_docs: list, web_results: list) -> str:
        """
        Build the LLM prompt context from retrieved sources.

        Args:
            query: Original user question.
            corpus_docs: Retrieved chunks from ChromaDB.
            web_results: Results from DuckDuckGo (may be empty).

        Returns:
            Formatted context string for the LLM.
        """
        return f"""
User Question: {query}

--- Internal Policy Documents ---
{json.dumps(corpus_docs, indent=2)}

--- Web Search Results ---
{json.dumps(web_results, indent=2) if web_results else "No web search performed."}

Instructions:
- Use internal documents as the primary source for policy facts.
- Use web results only for recent events or information not in the corpus.
- Clearly cite which source each fact comes from.
"""

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, query: str, on_step=None) -> dict:
        """
        Run the agent for a given user query.

        Args:
            query: The user's question.
            on_step: Optional callback function called after each step.
                     Receives a dict with keys: type, tool, inputs, result, etc.

        Returns:
            dict with keys:
                answer (str): Final answer from the LLM.
                steps (list): All intermediate steps taken.
                tokens_used (int): Total tokens consumed.
                iterations (int): Number of retrieval steps performed.
        """
        steps = []

        def emit(event: dict):
            """Record a step and optionally notify the caller."""
            steps.append(event)
            if on_step:
                on_step(event)

        emit({"type": "start", "query": query})

        # ── Stage 1: Corpus retrieval (always) ───────────────────────────────
        emit({
            "type": "tool_call",
            "tool": "retrieve_documents",
            "inputs": {"query": query},
            "iteration": 1
        })

        corpus_docs = self.store.retrieve(query, top_k=TOP_K)

        emit({
            "type": "tool_result",
            "tool": "retrieve_documents",
            "result": corpus_docs,
            "iteration": 1
        })

        # ── Stage 2: Web search (only for recent/current queries) ─────────────
        web_results = []
        iterations = 1

        if self._needs_web_search(query):
            iterations = 2
            emit({
                "type": "tool_call",
                "tool": "web_search",
                "inputs": {"query": query},
                "iteration": 2
            })

            web_results = self._web_search(query)

            emit({
                "type": "tool_result",
                "tool": "web_search",
                "result": web_results,
                "iteration": 2
            })

        # ── Stage 3: LLM synthesis ────────────────────────────────────────────
        context = self._build_context(query, corpus_docs, web_results)

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ],
            max_tokens=1024
        )

        answer = response.choices[0].message.content or ""
        emit({"type": "final_answer", "answer": answer})

        return {
            "answer": answer,
            "steps": steps,
            "tokens_used": response.usage.total_tokens,
            "iterations": iterations
        }