"""
RAG Pipeline Orchestration
===========================
Main pipeline that ties together all RAG components.
"""

import logging
from typing import List, Optional, Dict
from pathlib import Path

from .loader import load_documents, chunk_documents
from .embedder import get_embedder
from .retriever import create_vectorstore, get_retriever, load_vectorstore
from .generator import get_llm, create_handbook_prompt, create_qa_chain, generate_response


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG Pipeline class.

    """

    def __init__(
        self,
        data_dir: str = "data/",
        embedder_provider: str = "huggingface",
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "ollama",
        llm_model: str = "phi3",
        chunk_size: int = 800, #has been modified to match main.py
        chunk_overlap: int = 120, #has also been modified to match main.py
        retrieval_k: int = 4,
        vectorstore_type: str = "chroma",
        persist_dir: Optional[str] = "vectorstore",
        system_prompt: Optional[str] = None #for providing support to the system prompt

    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.persist_dir = persist_dir
        self.vectorstore_type = vectorstore_type
        self.system_prompt = system_prompt

        # Initializing components
        logger.info("Initializing embedder...")
        self.embedder = get_embedder(
            provider=embedder_provider,
            model_name=embedder_model
        )

        logger.info("Initializing LLM...")
        self.llm = get_llm(
            provider=llm_provider,
            model_name=llm_model,
            temperature=0.3 #made lower for more factual answers
        )

        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def load_and_index(self, force_rebuild: bool = False):
        """
        Load documents and create vector index.

        """
        # Check if vectorstore exists
        persist_path = Path(self.persist_dir) if self.persist_dir else None

        if persist_path and persist_path.exists() and not force_rebuild:
            logger.info("Loading existing vectorstore...")
            #Loading from persist_dir
            try:
                self.vectorstore = load_vectorstore(
                    self.embedder,
                    db_type=self.vectorstore_type,
                    persist_dir=self.persist_dir
                )
                logger.info("  ✓ Loaded existing vectorstore")
            except Exception as e:
                logger.warning(f"Could not load existing vectorstore: {e}")
                logger.info("Will rebuild from scratch...")
                self.vectorstore = None
        
        if self.vectorstore is None:
            logger.info("Loading and chunking documents...")
            documents = load_documents(self.data_dir)
            
            
            if not documents:
                raise FileNotFoundError(
                    f"No documents found in '{self.data_dir}'. "
                    f"Please add PDF, TXT, or MD files to this directory."
                )
            
            chunks = chunk_documents(
                documents,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info(f"  ✓ Created {len(chunks)} chunks from {len(documents)} documents")

            logger.info("Creating vector index...")
            self.vectorstore = create_vectorstore(
                chunks,
                self.embedder,
                db_type=self.vectorstore_type,
                persist_dir=self.persist_dir
            )

        logger.info("Setting up retriever...")
        self.retriever = get_retriever(
            self.vectorstore,
            k=self.retrieval_k
        )

        logger.info("Creating QA chain...")
        # Using handbook prompt with system prompt
        prompt = create_handbook_prompt(self.system_prompt)
        self.qa_chain = create_qa_chain(self.llm, self.retriever, prompt)

    def query(self, question: str, return_sources: bool = True) -> Dict:
        """
        Query the RAG pipeline.

        """
        if self.qa_chain is None:
            raise RuntimeError("Pipeline not initialized. Call load_and_index() first.")

        logger.info(f"Querying: {question[:50]}...")
        response = generate_response(self.qa_chain, question, return_sources)
        return response

    def evaluate(self, test_queries: List[Dict]) -> Dict:
        """
        Evaluating the pipeline on test queries.

        """
        results = []
        for item in test_queries:
            query = item.get["question"]
            expected = item.get("expected_answer", "")

            response = self.query(query, return_sources=True)
            results.append({
                "query": query,
                "expected": expected,
                "answer": response["answer"],
                "sources": response.get("sources", [])
            })

        # Implementing evaluation metrics

        return {"results": results}


def main():
    """Main entry point for testing."""
    pipeline = RAGPipeline(
        data_dir="data/",
        embedder_provider="huggingface",
        llm_provider="ollama",
        llm_model="phi3",
        retrieval_k=3
    )

    pipeline.load_and_index()

    # Example query
    response = pipeline.query("What is the academic integrity policy?") #modified for the handbook
    print("\nAnswer:", response["answer"])
    if "sources" in response:
        print("\nSources:", len(response["sources"]), "chunks retrieved")


if __name__ == "__main__":
    main()
