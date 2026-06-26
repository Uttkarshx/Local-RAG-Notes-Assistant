from pypdf import PdfReader 
from llama_index.core.node_parser import SentenceSplitter 
from sentence_transformers import SentenceTransformer 
import chromadb 
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.stdout.reconfigure(encoding="utf-8")

#Loading the pdf Documents which contains data  , and extracting into text string and then chunking the text into smaller chunks of 500 characters with an overlap of 50 characters.
reader = PdfReader("data/")
text = ""
for page in reader.pages : 
   page_text = page.extract_text()
   if page_text:
       text = text+page_text 

# CHunking process 
splitter = SentenceSplitter(
    chunk_size = 500 ,
    chunk_overlap = 50   
)

chunks = splitter.split_text(text) 

# Load EMbedding Model 
model = SentenceTransformer ("all-MiniLM-L6-v2")

#CHroma db store for storing the embeddings of the chunks and the question embedding for semantic search.
client =  chromadb.PersistentClient(
    path = "vector_store"
)

collection = client.get_or_create_collection(
    name = "attention_paper"
)


# Add all the chunk -> embeddings to the collection
for i , chunk in enumerate(chunks):
    embedding = model.encode(chunk)
    collection.add(
        ids = [str(i)],
        documents = [chunk],
        embeddings = [embedding.tolist()],
        metadatas = [{
            "page" : "unkown" , 
            "source" : "1706.03762v7.pdf"
            
        }
    ]
 )
print("Stored", len(chunks), "chunks successfully.")
