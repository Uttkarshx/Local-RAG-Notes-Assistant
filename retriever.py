import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model (only once)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="vector_store")

# Open existing collection
collection = client.get_collection(name="attention_paper")


def retrieve(question, n_results=3):
    """
    Retrieve top relevant chunks from ChromaDB.
    """

    # Convert question into embedding
    question_embedding = model.encode(question)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=n_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_chunks = []

    for i in range(len(documents)):

        retrieved_chunks.append(
            {
                "document": documents[i],
                "source": metadatas[i]["source"],
                "distance": distances[i]
            }
        )

    return retrieved_chunks


# Test only when running this file directly
if __name__ == "__main__":

    question = input("Ask Question: ")

    chunks = retrieve(question)

    print("\nRetrieved Chunks\n")

    for i, chunk in enumerate(chunks):

        print("=" * 60)
        print(f"Rank {i+1}")
        print("Distance :", chunk["distance"])
        print("Source   :", chunk["source"])
        print()
        print(chunk["document"])
        print()