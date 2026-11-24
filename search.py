
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# --------------------------------------------------
# 2. Initialize ChromaDB
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Create or retrieve collection
# We provide our own embeddings, so no default
# embedding function is needed.

collection = chroma_client.get_or_create_collection(
    name="programming_documents",
    embedding_function=None,
    metadata={"hnsw:space": "cosine"}
)

# --------------------------------------------------
# 3. Our document collection
# --------------------------------------------------

documents = [
    "Python is a popular programming language.",
    "FastAPI is a modern Python framework for building APIs.",
    "Machine learning enables computers to learn from data.",
    "Laravel is a PHP framework for web development.",
    "RAG allows AI applications to retrieve information from documents.",
]

document_ids = [
    "doc1",
    "doc2",
    "doc3",
    "doc4",
    "doc5",
]

# --------------------------------------------------
# 4. Generate document embeddings
# --------------------------------------------------

def generate_embeddings(texts):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    return [
        item.embedding
        for item in response.data
    ]


# --------------------------------------------------
# 5. Store documents and embeddings
# --------------------------------------------------

def store_documents():

    embeddings = generate_embeddings(documents)

    collection.upsert(
        ids=document_ids,
        documents=documents,
        embeddings=embeddings,
    )

    print(f"Stored documents: {collection.count()}")


# --------------------------------------------------
# 6. Semantic search
# --------------------------------------------------

def semantic_search(query, top_k=3):

    # Generate embedding for user query
    query_embedding = generate_embeddings([query])[0]

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"],
    )

    return results


# --------------------------------------------------
# 7. Display search results
# --------------------------------------------------

def display_results(results):

    documents_found = results["documents"][0]
    distances = results["distances"][0]

    print("\nSearch Results")
    print("=" * 50)

    for index, (document, distance) in enumerate(
        zip(documents_found, distances),
        start=1
    ):

        # ChromaDB returns cosine distance.
        # Convert it to cosine similarity.
        similarity = 1 - distance

        print(f"\nResult #{index}")
        print(f"Document: {document}")
        print(f"Similarity: {similarity:.4f}")
        print("-" * 50)


# --------------------------------------------------
# 8. Main application
# --------------------------------------------------

def main():

    print("Initializing Semantic Search Engine...")

    # Store documents in ChromaDB
    store_documents()

    print("\nSemantic Search Engine is ready!")

    while True:

        query = input(
            "\nEnter your search query (or 'exit'): "
        ).strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a search query.")
            continue

        results = semantic_search(
            query=query,
            top_k=3
        )

        display_results(results)


if __name__ == "__main__":
    main()
