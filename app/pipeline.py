# app/pipeline.py
import os, pickle
import faiss
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Initialize once
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(text: str, sid: str):
    # 1. Chunk
    chunks = splitter.split_text(text)
    # 2. Embed
    embeddings = embed_model.encode(chunks)
    # 3. Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    # 4. Persist
    session_dir = f"./data/{sid}"
    os.makedirs(session_dir, exist_ok=True)
    faiss.write_index(index, f"{session_dir}/faiss.index")
    with open(f"{session_dir}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_index_and_chunks(sid: str):
    session_dir = f"./data/{sid}"
    index = faiss.read_index(f"{session_dir}/faiss.index")
    with open(f"{session_dir}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks
