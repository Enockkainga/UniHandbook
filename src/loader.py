"""
Document Loader and Chunker
===========================
"""

from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader
)
#changing
from langchain_text_splitters import (RecursiveCharacterTextSplitter)
from langchain_core.documents import Document


def load_documents(data_dir: str = "data/") -> List[Document]:
    """
    Load documents from the data directory.
    Supports PDF, TXT and MD files.
    """
    path = Path(data_dir)

    if not path.exists():
        raise FileNotFoundError(f"Directory '{data_dir}' does not exist")

    documents = []

     # Load PDF files (ENABLED - critical for handbooks)
    for pdf_path in path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = pdf_path.name
                doc.metadata["type"] = "pdf"
            documents.extend(docs)
            print(f"  ✓ Loaded PDF: {pdf_path.name}")
        except Exception as e:
            print(f"  ✗ Error loading {pdf_path.name}: {e}")
    

    # Basic text file loading - extend this for your scenario
    for file_path in path.glob("*.txt"):
        
        try:
            #changing
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["type"] = "text"
            documents.extend(docs)
            print(f"  ✓ Loaded text: {file_path.name}")
        except Exception as e:
            print(f"  ✗ Error loading {file_path.name}: {e}")


    # Load Markdown files
    for md_path in path.glob("*.md"):
        try:
            loader = TextLoader(str(md_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = md_path.name
                doc.metadata["type"] = "markdown"
            documents.extend(docs)
            print(f"  ✓ Loaded markdown: {md_path.name}")
        except Exception as e:
            print(f"  ✗ Error loading {md_path.name}: {e}")
    
    return documents




def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800, #matching with main.py
    chunk_overlap: int = 120, #also matching with main.py
    chunking_strategy: str = "recursive"
) -> List[Document]:
    """
    Splitting documents into chunks.

    """
    if chunking_strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )

    else:
        print(f"  ⚠ Unknown strategy '{chunking_strategy}', falling back to recursive")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    chunks = splitter.split_documents(documents)


    # Add chunk metadata for better tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown"
    
    print(f"  ✓ Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


if __name__ == "__main__":
    # Test the loader
    try:
        docs = load_documents("data/")
        if docs:
            chunks = chunk_documents(docs)
            print(f"\nFirst chunk preview:")
            print(f"Source: {chunks[0].metadata.get('source', 'unknown')}")
            print(f"Content: {chunks[0].page_content[:150]}...")
        else:
            print("No documents found. Add PDF/TXT files to 'data/' directory.")
    except FileNotFoundError as e:
        print(f"Error: {e}")