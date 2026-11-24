from pdf_reader import client, embedding, get_collection

CHAT_MODEL = "gpt-4o-mini"

collection = get_collection()


# --------------------------------------------------
# 1. RETRIEVE: find the chunks most related to the question
# --------------------------------------------------

def retrieve(question, top_k=3):
    results = collection.query(
        query_embeddings=embedding([question]),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {"text": text, "source": meta["source"], "page": meta["page"], "score": 1 - dist}
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


# --------------------------------------------------
# 2. AUGMENT: put the retrieved chunks into the prompt
# --------------------------------------------------

def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[{i}] ({c['source']}, page {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )

    return (
        "Answer the question using ONLY the context below.\n"
        "Cite the sources you used like [1] or [2].\n"
        "If the context does not contain the answer, "
        "say \"I don't know based on the provided documents.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


# --------------------------------------------------
# 3. GENERATE: let the LLM write the answer
# --------------------------------------------------

def answer(question, top_k=3):
    chunks = retrieve(question, top_k)
    prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content, chunks


def main():
    if collection.count() == 0:
        print("No documents found. Run: python pdf_reader.py nimbusdb_handbook.pdf")
        return

    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()

        if question.lower() == "exit":
            break
        if not question:
            continue

        reply, chunks = answer(question)

        print("\nRetrieved context:")
        for i, c in enumerate(chunks, start=1):
            print(f"  [{i}] {c['source']} p.{c['page']} (similarity {c['score']:.3f})")
            print(f"      {c['text'][:100]}...")

        print(f"\nAnswer: {reply}")


if __name__ == "__main__":
    main()
