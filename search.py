from sentence_transformers import SentenceTransformer
import faiss
import pickle
import pickle

index = faiss.read_index('faiss_index.bin')

with open('contents_db.pkl', 'rb') as f:
    contents_db = pickle.load(f)

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Query example
query = "Địa chỉ quán ốc"
query_embedding = model.encode([query]).astype('float32') 
D, I = index.search(query_embedding, k=5)  

# Print the most similar contents
for idx in I[0]:
    print(contents_db[idx])