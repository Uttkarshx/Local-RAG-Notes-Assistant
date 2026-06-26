from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()

for doc in documents:
    print("\nFile Name:", doc.metadata.get("file_name"))
    print("-" * 50)
    print(doc.text[:300])