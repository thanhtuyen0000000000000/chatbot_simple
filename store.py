import pandas as pd
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, CollectionConfig
from qdrant_client.http.models import VectorParams, Distance, OptimizersConfigDiff
from qdrant_client.http.models import PointStruct
import numpy as np
import os
# model = SentenceTransformer('dangvantuan/vietnamese-embedding')
# model.save(r"D:\UIT\Chatbot HCM")

model = SentenceTransformer(r"D:\UIT\Chatbot HCM\models")

df = pd.read_csv(r"D:\UIT\Chatbot HCM\chunked_province_websites_content.csv")


 
qdrant_url = "https://23d8a1f2-be16-416f-9b2b-86f775e12352.us-east4-0.gcp.cloud.qdrant.io:6333"  # Replace with your Qdrant Cloud URL
api_key = "1MmduvuJxAzGCImQpHEwuCy18La4u8gPTwH0AQm26nd-eOrzXImJgw"  # Replace with your Qdrant API Key

# # Initialize QdrantClient for Cloud
client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=10000)
# client = QdrantClient(":memory:")
collection_name = "province_embeddings"
# Check if the collection exists, if not, create it
try:
    client.get_collection(collection_name)
except Exception as e:
    print(f"Collection not found. Creating collection: {e}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
        size=model.get_sentence_embedding_dimension(),  # Kích thước vector
        distance=Distance.COSINE  # Loại khoảng cách để tính toán tương đồng
    ),
    )
print("Create collection success!")
# Step 5: Process each row sequentially
for i, row in df.iterrows():
    # Tokenize and encode the 'Content' of each row
    tokenized_content = tokenize(row['Content'])
    embedding = model.encode([tokenized_content])

    # Prepare data for Qdrant insertion (ID, vector, and metadata)
    point_id = i  # Unique ID for each embedding
    vector = embedding.tolist()  # Convert embedding to a list
    metadata = {"province": row['province'], "url": row['url']}

    # Step 6: Insert embedding into Qdrant
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload=metadata
            )
        ]
    )

print("Sequentially inserted embeddings into Qdrant, and collection is stored locally.")

