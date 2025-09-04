import sys, chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from services.llm_adapter import generate_with_llama2
import config

def main():
    if len(sys.argv)<2:
        print("Usage: python services/rag_query.py \"your question\""); return
    q = sys.argv[1]
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    coll = client.get_or_create_collection("events", embedding_function=ef)
    res = coll.query(query_texts=[q], n_results=5)
    docs = res.get("documents",[[]])[0]; metas = res.get("metadatas",[[]])[0]
    ctx = "".join([f"- {d} [{m}]\n" for d,m in zip(docs, metas)])
    if not ctx: print("No relevant logs found."); return
    print("Top matches:\n", ctx)
    ans = generate_with_llama2(f"Use ONLY this context:\n{ctx}\n\nQ: {q}\nA:")
    if ans: print("\nLLM answer:\n", ans)
    else: print("\nExtractive answer:\n", docs[0] if docs else "N/A")

if __name__ == "__main__":
    main()
