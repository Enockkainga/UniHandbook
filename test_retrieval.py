from src.retriever import load_vectorstore
from src.embedder import get_embedder

# Load everything
print("Loading embedder...")
embedder = get_embedder()

print("Loading vectorstore...")
# Use the correct path - your vectorstore is in the Demo folder
vectorstore = load_vectorstore(embedder, persist_dir="./vectorstore")

# Test query
query = "What is the GPA requirement for undergraduate admission?"
print(f"\nQuery: {query}\n")
print("=" * 50)

# Search for relevant documents
results = vectorstore.similarity_search_with_relevance_scores(query, k=5)

print(f"Found {len(results)} results:\n")
for i, (doc, score) in enumerate(results):
    print(f"--- Result {i+1} (Relevance Score: {score:.3f}) ---")
    print(doc.page_content[:400])
    print()