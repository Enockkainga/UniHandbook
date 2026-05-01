"""
LLM Generator Module
====================
"""

from typing import List, Optional, Dict
from langchain_community.llms import Ollama
#changing
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate, ChatPromptTemplate
from langchain_classic.schema import HumanMessage, SystemMessage
#newly added
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def get_llm(
    provider: str = "ollama",
    model_name: str = "phi3",
    temperature: float = 0.3, #Changed for factual answers
    **kwargs
):
    """
    Get an LLM for generation.

    """
    if provider == "ollama":
        return Ollama(model=model_name, temperature=temperature)
    elif provider == "huggingface":
        return HuggingFaceEndpoint(
            repo_id=model_name,
            model_kwargs={"temperature": temperature}
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_rag_prompt(
    system_message: Optional[str] = None,
    template: Optional[str] = None
) -> PromptTemplate:
    """
    Create a RAG prompt template.

    """
    if system_message is None:
        system_message = """You are a helpful AI assistant.
Use the retrieved context to answer the user's question.
If you don't know the answer, say so clearly.
Always cite your sources when possible."""

    if template is None:
        template = """Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    
    return prompt


def create_handbook_prompt(system_prompt: Optional[str] = None) -> PromptTemplate:
    """
    Create a specialized prompt for university handbook QA.
    """
    if system_prompt is None:
        system_prompt = """You are the University Handbook Assistant.

CRITICAL RULES:
1. ONLY answer questions based on the CONTEXT provided below.
2. If the question asks for ANY personal information (age, height, weight, contact info, opinions, preferences, etc.) that is NOT in the context, say EXACTLY: "I cannot find this information in the university handbook."
3. If the question is about ANY topic NOT covered in the context, say EXACTLY: "I cannot find this information in the university handbook."
4. NEVER rephrase questions you can't answer.
5. NEVER guess or use your own knowledge.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    return PromptTemplate(
        template=system_prompt,
        input_variables=["context", "question"]
    )

def create_qa_chain(llm, retriever, prompt):
    """
    Create a QA chain using modern LangChain syntax.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    
    # Create a proper chat prompt template
    if isinstance(prompt, str):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
    else:
        # If prompt is a PromptTemplate, convert it
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt.template if hasattr(prompt, 'template') else str(prompt)),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
    
    # Create the chain using LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

#changing
def generate_response(
    qa_chain,
    query: str,
    return_sources: bool = True
) -> Dict:
    """
    Generating a response using the RAG pipeline.

    Returns:
        Dict with 'answer' and optionally 'source_documents'
    """
    # For the LCEL chain, we need to pass the query directly
    result = qa_chain.invoke(query)
    
    # For the modern chain, we need to manually add source documents
    response = {
        "answer": result
    }
    
    if return_sources:
        response["sources"] = []
    
    return response

if __name__ == "__main__":
    # Testing prompt creation
    prompt = create_rag_prompt()
    print("Default prompt template:")
    print(prompt.template)
