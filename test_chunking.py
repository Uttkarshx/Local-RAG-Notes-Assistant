from pypdf import PdfReader
from llama_index.core.node_parser import SentenceSplitter

# Read PDF
reader = PdfReader("data/1706.03762v7.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

print("Total Characters:", len(text))

# Chunking
splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=50
)
    
chunks = splitter.split_text(text)

print("Total Chunks:", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0])