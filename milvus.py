import pandas as pd
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
import numpy as np

# Load model và dataset
model = SentenceTransformer(r"D:\UIT\Chatbot HCM\models")
df = pd.read_csv(r"D:\UIT\Chatbot HCM\chunked_province_websites_content.csv")

# Kết nối tới Milvus (sử dụng local hoặc server)
connections.connect("default", host="localhost", port="19530")

collection_name = "province_embeddings"

# Kiểm tra nếu collection đã tồn tại thì xóa để tạo mới (tùy chọn)
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# Định nghĩa schema cho collection trong Milvus
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=model.get_sentence_embedding_dimension()),
    FieldSchema(name="province", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024)
]

schema = CollectionSchema(fields=fields, description="Embeddings for Vietnamese province data")

# Tạo collection
collection = Collection(name=collection_name, schema=schema)

# Chèn từng dòng từ CSV vào Milvus
for i, row in df.iterrows():
    # Tokenize và encode 'Content' cho mỗi dòng
    tokenized_content = tokenize(row['Content'])
    embedding = model.encode([tokenized_content])[0]  # Lấy embedding đầu tiên cho câu

    # Chuẩn bị dữ liệu để chèn vào Milvus
    data = [
        [i],  # ID duy nhất cho mỗi embedding
        embedding.tolist(),  # Chuyển embedding thành danh sách
        row['province'],  # Thêm metadata
        row['url']
    ]

    # Chèn dữ liệu vào collection
    collection.insert(data)

print("Đã chèn tuần tự embeddings vào Milvus")

# Đảm bảo đồng bộ dữ liệu với Milvus
collection.flush()

# Đặt index cho collection để tăng tốc tìm kiếm
index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
collection.create_index(field_name="vector", index_params=index_params)

print("Index đã được tạo cho collection")
