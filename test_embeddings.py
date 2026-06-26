from pypdf import PdfReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Read PDF
reader = PdfReader("data/1706.03762v7.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

# Chunking
splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Total Chunks:", len(chunks))

# Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert first chunk into vector
embedding = model.encode(chunks[0])

print("Vector Length:", len(embedding))

print("\nFirst 10 Values:")
print(embedding[:10])
question = "What is self-attention?"
question_embedding = model.encode(question)

similarity = cosine_similarity(
    [question_embedding],
    [embedding]
)

print("Similarity:" ,similarity)
