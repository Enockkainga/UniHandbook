"""RAG Starter Source Package"""
# src/__init__.py
# Makes src a proper Python package

from .pipeline import RAGPipeline
from .loader import load_documents, chunk_documents
from .embedder import get_embedder
from .retriever import create_vectorstore, get_retriever
from .generator import get_llm, create_qa_chain, generate_response

__all__ = [
    'RAGPipeline',
    'load_documents',
    'chunk_documents',
    'get_embedder',
    'create_vectorstore',
    'get_retriever',
    'get_llm',
    'create_qa_chain',
    'generate_response',
]