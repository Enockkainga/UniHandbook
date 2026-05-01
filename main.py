"""
NOTE!!!....TINYLLAMA WAS USED INSTEAD OF THE REQUIRED NORMAL OLLAMA 
SINCE THE MACHINE THAT WAS USED LACKED THE COMPUTING POWER AND RESOURCES

"""


"""
University Handbook Assistant - CLI Interface
==============================================
Interactive RAG assistant that answers student questions about
university policies, procedures, and academic regulations using
the official university handbook(s).
"""

import argparse
import sys
from src.pipeline import RAGPipeline



# System prompt tailored for a university handbook assistant.
# The pipeline can pass this to the LLM (if supported) so answers
# stay grounded, cite policies, and avoid hallucinating rules.
HANDBOOK_SYSTEM_PROMPT = (
    "You are the University Handbook Assistant. You help students, faculty, "
    "and staff understand university policies, academic regulations, "
    "registration procedures, grading, code of conduct, financial aid, "
    "and student services. "
    "Answer ONLY using the provided handbook excerpts. "
    "If the handbook does not cover the question, say so clearly and "
    "suggest contacting the relevant office (e.g., Registrar, Dean of "
    "Students, Financial Aid). "
    "Always cite the handbook section or page when possible. "
    "Be concise, accurate, and use a friendly, supportive tone."
)


def print_banner(subtitle: str):
    print("=" * 60)
    print("  🎓 University Handbook Assistant")
    print(f"  {subtitle}")
    print("=" * 60)


def interactive_mode(pipeline: RAGPipeline):
    """Run the handbook assistant in interactive mode."""
    print_banner("Interactive Mode")
    print("Ask any question about university policies and procedures.")
    print("Examples:")
    print("  • What is the policy on academic dishonesty?")
    print("  • How do I withdraw from a course?")
    print("  • What are the graduation requirements?")
    print("Type 'quit', 'exit', or 'q' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye! Good luck with your studies. 🎓")
                break

            if not query:
                continue

            response = pipeline.query(query, return_sources=True)

            print(f"\nAssistant: {response['answer']}")

            if "sources" in response and response["sources"]:
                print("\n📚 Handbook references:")
                for i, source in enumerate(response["sources"], 1):
                    meta = source.get("metadata", {})
                    src_name = meta.get("source", "Unknown handbook")
                    section = meta.get("section") or meta.get("page")
                    label = f"{src_name}" + (f" — {section}" if section else "")
                    print(f"  {i}. {label}")
                    print(f"     {source['content'][:120].strip()}...")

            print()

        except KeyboardInterrupt:
            print("\nGoodbye! Good luck with your studies. 🎓")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def demo_mode(pipeline: RAGPipeline):
    """Run with predefined demo questions about a typical handbook."""
    demo_questions = [
        "What is the university's policy on academic integrity?",
        "How do I add or drop a course after the semester begins?",
        "What are the requirements to graduate with honors?",
        "What is the attendance policy for undergraduate courses?",
        "How do I appeal a final grade?",
        "What student support services are available?",
    ]

    print_banner("Demo Mode")
    print("Running sample questions a student might ask...\n")

    for question in demo_questions:
        print(f"Q: {question}")
        response = pipeline.query(question)
        print(f"A: {response['answer']}\n")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="University Handbook Assistant - chat with your university handbook"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "demo"],
        default="interactive",
        help="Mode to run the assistant",
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Directory containing handbook documents (PDF, TXT, MD, HTML)",
    )
    parser.add_argument(
        "--embedder",
        default="huggingface",
        choices=["huggingface", "openai", "ollama"],
        help="Embedder provider",
    )
    parser.add_argument(
        "--llm",
        default="ollama",
        choices=["ollama", "huggingface", "openai"],
        help="LLM provider",
    )
    parser.add_argument(
        "--llm-model",
        default="phi3",
        help="LLM model name",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of handbook chunks to retrieve per question",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size for splitting handbook text "
             "(handbooks read better with larger chunks)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="Overlap between chunks to preserve policy context",
    )

    args = parser.parse_args()

    print("Initializing University Handbook Assistant...")
    print(f"  📂 Handbook directory: {args.data_dir}")
    print(f"  🔎 Embedder: {args.embedder}")
    print(f"  🤖 LLM: {args.llm} ({args.llm_model})")
    print(f"  📑 Retrieval K: {args.k}")
    print(f"  ✂️  Chunk size / overlap: {args.chunk_size} / {args.chunk_overlap}")
    print()

    # Build pipeline. We try to pass the handbook system prompt if the
    # pipeline accepts it; otherwise we fall back gracefully.
    pipeline_kwargs = dict(
        data_dir=args.data_dir,
        embedder_provider=args.embedder,
        llm_provider=args.llm,
        llm_model=args.llm_model,
        retrieval_k=args.k,
        chunk_size=args.chunk_size,
    )
    try:
        pipeline = RAGPipeline(
            **pipeline_kwargs,
            chunk_overlap=args.chunk_overlap,
            system_prompt=HANDBOOK_SYSTEM_PROMPT,
        )
    except TypeError:
        # Older pipeline signature — keep it working.
        pipeline = RAGPipeline(**pipeline_kwargs)

    try:
        pipeline.load_and_index()
    except FileNotFoundError:
        print(
            f"\n⚠️  No handbook documents found in '{args.data_dir}'.\n"
            "   Place your university handbook (PDF / TXT / MD) inside that folder\n"
            "   and run the assistant again."
        )
        sys.exit(1)

    if args.mode == "interactive":
        interactive_mode(pipeline)
    else:
        demo_mode(pipeline)


if __name__ == "__main__":
    main()
