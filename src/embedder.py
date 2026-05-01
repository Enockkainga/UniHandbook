"""
Embedding Model
===============
Students must choose and configure the embedding model.
"""

from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


def get_embedder(
    provider: str = "huggingface",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs
):
    """
    Getting an embedding model.

    """
    if provider == "huggingface":
        # Open-source, free to use
        #Auto-detecting GPU
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )
    elif provider == "openai":
        # Requires OPENAI_API_KEY
        # Explore the following link for free usable api keys:
        # https://console.groq.com/keys
        return OpenAIEmbeddings(model=model_name)
    elif provider == "ollama":
        # Local inferencing via Ollama
        return OllamaEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def embed_documents(embedder, documents: List) -> List[List[float]]:
    """
    Embedding a list of documents.

    Adding batch processing for large document sets
    """

    #handling both document objects and plain strings
    if documents and hasattr(documents[0], 'page_content'):
       texts = [doc.page_content for doc in documents]
    
    else:
        texts = documents
    
    # Process in batches to avoid memory issues with large document sets
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


def embed_query(embedder, query: str) -> List[float]:
    """
    Embeding a query string.
    """
    return embedder.embed_query(query)


if __name__ == "__main__":
    print("Testing embedding model...")
    embedder = get_embedder()
    test_text = "What is Retrieval-Augmented Generation?"
    embedding = embed_query(embedder, test_text)
    print(f"Embedding dimension: {len(embedding)}")
#changing
#Testing batch embedding
#test_docs = ["Document 1", "Document 2", "Document 3"]
#embeddings = embed_documents(embedder, test_docs)
#print(f"Batch embedded {len(embeddings)} documents")