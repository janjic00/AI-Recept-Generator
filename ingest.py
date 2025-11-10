import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# ----------------------------
# 1. Load API keys
# ----------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "cookbook")

# ----------------------------
# 2. Initialize Pinecone
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# If the index does not exist, create a new one
if PINECONE_INDEX not in pc.list_indexes().names():
    print(f"✅ Index '{PINECONE_INDEX}' does not exist. Creating new index...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,  # for all-MiniLM-L6-v2 embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"⚠️ Index '{PINECONE_INDEX}' already exists. Using existing index...")

index = pc.Index(PINECONE_INDEX)

# ----------------------------
# 3. Load knowledge base
# ----------------------------
with open("knowledgebase.json", "r", encoding="utf-8") as f:
    knowledge = json.load(f)

# ----------------------------
# 4. Initialize embedding model
# ----------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------
# 5. Upsert embeddings into Pinecone
# ----------------------------
print("📤 Upserting embeddings into Pinecone...")
for recipe in knowledge:
    text = recipe.get("title", "") + " " + recipe.get("ingredients", "") + " " + recipe.get("instructions", "")
    embedding = embed_model.encode(text).tolist()
    index.upsert([(recipe["id"], embedding, recipe)])

print("✅ Knowledge base has been inserted/updated in Pinecone!")
