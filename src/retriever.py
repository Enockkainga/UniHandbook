"""
Retriever Module
================
Students must implement and customize retrieval strategies.
"""
#changing from old structre
from typing import List, Optional, Dict
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo


def create_vectorstore(
    documents: List[Document],
    embedder,
    db_type: str = "chroma",
    persist_dir: Optional[str] = None
):
    """
    Create a vector store from documents.

    """
    if db_type == "chroma":
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedder,
            persist_directory=persist_dir
        )
        #Persisting database to the disk
        if persist_dir:
            vectorstore.persist()
            print(f"  ✓ ChromaDB persisted to {persist_dir}")

    elif db_type == "faiss":
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedder
        )
        if persist_dir:
            vectorstore.save_local(persist_dir)
            print(f"  ✓ FAISS index saved to {persist_dir}")
    else:
        raise ValueError(f"Unknown DB type: {db_type}")

    
    return vectorstore


def load_vectorstore(
    embedder,
    db_type: str = "chroma",
    persist_dir: Optional[str] = None
):
    """
    Load an existing vector store from disk.
    Useful for skipping re-indexing on subsequent runs.
    """
    if not persist_dir:
        raise ValueError("persist_dir required to load existing vectorstore")
    
    if db_type == "chroma":
        return Chroma(
            embedding_function=embedder,
            persist_directory=persist_dir
        )
    elif db_type == "faiss":
        return FAISS.load_local(persist_dir, embedder, allow_dangerous_deserialization=True)
    else:
        raise ValueError(f"Unknown DB type: {db_type}")


def get_retriever(
    vectorstore,
    search_type: str = "similarity",
    k: int = 4,
    score_threshold: Optional[float] = None,
    filter_criteria: Optional[Dict] = None
):
    """
    Create a retriever with customizable search parameters.

    """
    search_kwargs = {"k": k}

    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    if filter_criteria is not None:
        search_kwargs["filter"] = filter_criteria

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )

    return retriever


def retrieve_with_hybrid_search(
    vectorstore,
    query: str,
    k: int = 4,
    alpha: float = 0.5
) -> List[Document]:
    """
    Implementing hybrid search combining BM25 and vector similarity.

    """
    raise NotImplementedError("Hybrid search not yet implemented")


def retrieve_with_reranking(
    retriever,
    query: str,
    k: int = 4
) -> List[Document]:
    """
    Implementing reranking for improved relevance.

    """
    raise NotImplementedError("Reranking not yet implemented")


if __name__ == "__main__":
    # Basic test
    
    import sys
    from pathlib import Path
    
    # Add current directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    
    from embedder import get_embedder
    from loader import load_documents, chunk_documents

    print("Testing retriever module...")
    
    try:
        docs = load_documents("data/")
        if not docs:
            print("No documents found. Create a test file in data/ directory.")
        else:
            chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
            embedder = get_embedder()
            
            vectorstore = create_vectorstore(chunks, embedder, persist_dir="test_vectorstore")
            retriever = get_retriever(vectorstore, k=3)
            
            results = retriever.invoke("What is the academic policy?")
            print(f"Retrieved {len(results)} documents")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")