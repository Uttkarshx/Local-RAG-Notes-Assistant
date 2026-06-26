from retriever import retrieve
from generator import generate


question = input("Ask Question: ")

chunks = retrieve(question)

context = ""

for chunk in chunks:
    context += chunk["document"]
    context += "\n\n"

prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context.

If the answer is not contained in the context, reply exactly:

"I couldn't find this information in the uploaded documents."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:


"""

answer = generate(prompt)

print("\nAnswer:\n")
print(answer)