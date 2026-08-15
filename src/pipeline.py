import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm():
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(vector_store, llm, question: str) -> dict[str, str | list[str]]:
    """Retrieve relevant chunks and generate an answer.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string
    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """

    docs = vector_store.similarity_search(question, k=3)
    sources: list[str] = [doc.page_content for doc in docs]

    context = "\n\n".join(sources)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    result = llm(prompt)
    answer = result[0]["generated_text"]

    if not answer.strip():
        answer = "No answer was generated for that question."

    return {"answer": answer, "sources": sources}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Interactive Q&A loop.
    """
    query: str | None = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--query" else None

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    #Error handling for missing file
    if not os.path.isdir(data_dir):
        sys.exit(f"No data directory found at {os.path.abspath(data_dir)}")
    if not any(f.endswith(".txt") for _, _, files in os.walk(data_dir) for f in files):
        sys.exit(f"No .txt documents found in {os.path.abspath(data_dir)}")

    vector_store = build_knowledge_base(data_dir)
    llm = get_llm()

    if not query:
        print("Ask a question about our services, pricing, or process.")
        print("Type 'quit' to exit.\n")

    while True:
        question = query or input("> ").strip()
        if question == "quit":
            break
        if not question:
            continue

        result = ask_question(vector_store, llm, question)

        print("\nSources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"  {i}. {' '.join(source.split())}")
        print(f"\nAnswer: {result['answer']}\n")

        if query:
            break


if __name__ == "__main__":
    main()