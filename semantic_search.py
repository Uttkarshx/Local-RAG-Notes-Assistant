from pypdf import PdfReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys

sys.stdout.reconfigure(encoding="utf-8")

# =====================================
# STEP 1: LOAD PDF
# =====================================

reader = PdfReader("data/1706.03762v7.pdf")

text = ""

for page in reader.pages:
    extracted_text = page.extract_text()

    if extracted_text:
        text += extracted_text

print("Total Characters:", len(text))

# =====================================
# STEP 2: CHUNKING
# =====================================

splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Total Chunks:", len(chunks))

# =====================================
# STEP 3: LOAD EMBEDDING MODEL
# =====================================

model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================
# STEP 4: USER QUESTION
# =====================================

question = "What is self-attention?"

print("\nQuestion:", question)

question_embedding = model.encode(question)

# =====================================
# STEP 5: SEMANTIC SEARCH
# =====================================

scores = []

for chunk in chunks:

    chunk_embedding = model.encode(chunk)

    similarity_score = cosine_similarity(
        [question_embedding],
        [chunk_embedding]
    )[0][0]

    scores.append(
        (similarity_score, chunk)
    )

# =====================================
# STEP 6: SORT BY SIMILARITY
# =====================================

scores.sort(
    key=lambda x: x[0],
    reverse=True
)

# =====================================
# STEP 7: SHOW TOP 3 RESULTS
# =====================================

print("\n" + "="*60)
print("TOP 3 MOST RELEVANT CHUNKS")
print("="*60)

for i in range(3):

    score = scores[i][0]
    chunk = scores[i][1]

    print(f"\nRank #{i+1}")
    print(f"Similarity Score: {score:.4f}")

    print("\nChunk Preview:")
    print(chunk[:500])

    print("\n" + "-"*60)
    

    