from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from nltk.tokenize import sent_tokenize
import pandas as pd

# Bước 1: Khởi tạo model và FAISS
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
dimension = 384  # Số chiều của embedding
index = faiss.IndexFlatL2(dimension)  # Khởi tạo FAISS index với L2 (Euclidean distance)

# Đường dẫn tới file CSV

file_path = r'D:\UIT\Chatbot HCM\chunked_province_websites_content.csv'
df = pd.read_csv(file_path)

sentences_db = []  # Danh sách để lưu lại các câu từ mỗi dòng của Content

# Bước 2: Đọc từng dòng của DataFrame và xử lý
for i, row in df.iterrows():
    content = row["Content"]  # Lấy nội dung của cột "Content"
    sentences = sent_tokenize(content)  # Tách câu trong nội dung
    
    embeddings = model.encode(sentences)  # Tính toán embedding cho mỗi câu
    faiss_embeddings = np.array(embeddings).astype('float32')  # Chuyển sang định dạng float32 cho FAISS
    index.add(faiss_embeddings)  # Thêm embedding vào FAISS index
    sentences_db.extend(sentences)  # Lưu các câu để sử dụng khi truy vấn

# Bước 3: Truy vấn (Ví dụ)
query = "thời tiết"
query_embedding = model.encode([query]).astype('float32')  # Tính embedding cho câu truy vấn
D, I = index.search(query_embedding, k=5)  # k là số kết quả gần nhất muốn lấy

# In ra các câu tương tự nhất
for idx in I[0]:  # I là mảng 2D nên lấy I[0]
    print(sentences_db[idx])
