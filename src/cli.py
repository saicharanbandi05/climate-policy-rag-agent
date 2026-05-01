import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from vectorstore import VectorStore
from agent import ClimateAgent

def on_step(event):
    if event["type"] == "tool_call":
        print(f"\n  Searching: '{event['inputs'].get('query', '')}'")
    elif event["type"] == "tool_result":
        result = event["result"]
        if isinstance(result, list):
            for r in result[:2]:
                if isinstance(r, dict) and "doc_title" in r:
                    print(f"  Found: [{r.get('score','')}] {r['doc_title']}")

def main():
    print("Loading vector store...")
    store = VectorStore()
    agent = ClimateAgent(store)

    print("\n" + "="*50)
    print("  Climate Policy RAG Agent")
    print("  Type 'quit' to exit")
    print("="*50 + "\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nThinking...\n")
        result = agent.run(query, on_step=on_step)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\n[{result['iterations']} iterations | {result['tokens_used']} tokens]\n")
        print("-"*50)

if __name__ == "__main__":
    main()