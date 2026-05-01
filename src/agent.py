import json
import os
from groq import Groq
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": "Search the internal climate policy corpus. Use this FIRST for any climate policy question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant policy documents."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information not in the corpus. Use when the question is about recent events, news, or topics outside the policy documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Web search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List all available documents in the corpus.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

SYSTEM_PROMPT = """You are a powerful climate policy analyst assistant with two knowledge sources:

1. INTERNAL CORPUS - 10 authoritative climate policy documents (Paris Agreement, IRA, IPCC AR6, EU Green Deal, Carbon Pricing, Environmental Justice, Renewable Energy, US NCA, Methane Policy, Climate Finance)

2. WEB SEARCH - for current events, recent news, or topics outside the corpus

## Rules
- For climate policy questions: ALWAYS call retrieve_documents first
- If corpus results are insufficient: use web_search to supplement
- For non-climate questions or recent events: use web_search directly
- Always cite your sources (document title or website)
- Be precise with numbers, dates, and percentages
- If combining corpus + web results, clearly distinguish the sources"""

class ClimateAgent:
    def __init__(self, vector_store):
        self.store = vector_store
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def _web_search(self, query):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=4))
            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                })
            return formatted
        except Exception as e:
            return [{"error": f"Web search failed: {str(e)}"}]

    def _execute_tool(self, name, inputs):
        if name == "retrieve_documents":
            results = self.store.retrieve(inputs["query"], top_k=5)
            return json.dumps(results, indent=2)
        elif name == "web_search":
            results = self._web_search(inputs["query"])
            return json.dumps(results, indent=2)
        elif name == "list_documents":
            with open("data/documents.json", encoding="utf-8") as f:
                docs = json.load(f)
            return json.dumps([{"id": d["id"], "title": d["title"],
                                "source": d["source"]} for d in docs])
        return json.dumps({"error": f"Unknown tool: {name}"})

    def run(self, query, on_step=None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

        steps = []
        total_tokens = 0
        iterations = 0
        final_answer = ""

        def emit(event):
            steps.append(event)
            if on_step:
                on_step(event)

        emit({"type": "start", "query": query})
                # STEP 1: retrieve from corpus
        emit({
            "type": "tool_call",
            "tool": "retrieve_documents",
            "inputs": {"query": query},
            "iteration": 1
        })

        docs = self.store.retrieve(query, top_k=5)

        emit({
            "type": "tool_result",
            "tool": "retrieve_documents",
            "result": docs,
            "iteration": 1
        })

        # STEP 2: optional web search
        recent_keywords = ["latest", "today", "recent", "current", "news", "summit"]

        web_results = []
        if any(k in query.lower() for k in recent_keywords):
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

        context = f"""
User Question: {query}

Internal Documents:
{json.dumps(docs, indent=2)}

Web Results:
{json.dumps(web_results, indent=2)}

Instructions:
- Use internal documents when possible
- Use web results for recent info
- Always cite sources
"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
            "iterations": 1
        }