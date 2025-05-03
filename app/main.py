# app/main.py

# Load environment variables from .env before any os.getenv usage
from dotenv import load_dotenv
load_dotenv()

import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .ocr import pdf_to_text, image_to_text
from .pipeline import build_faiss_index, load_index_and_chunks, embed_model

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings  # <— added

# Prompt template for QA
template = """
You are a medical‐document summarization assistant. Use the excerpts below to answer the user’s question, but always

  •  Provide your response as concise bullet points (each starting with “- ”)  
  •  Focus on the most important findings or instructions  
  •  If the information isn’t in the text, reply “I don’t know.”

Context:
{context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(
    input_variables=["context","question"],
    template=template
)

app = FastAPI()

# Enable CORS so your frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1) Upload
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    sid = str(uuid.uuid4())
    raw_dir = f"./data/{sid}/raw"
    os.makedirs(raw_dir, exist_ok=True)
    dest = os.path.join(raw_dir, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"session_id": sid}

# 2) Process
@app.post("/process/{sid}")
def process(sid: str):
    raw_dir = f"./data/{sid}/raw"
    if not os.path.exists(raw_dir):
        raise HTTPException(404, "Session not found")
    path = next(os.scandir(raw_dir)).path
    if path.lower().endswith((".png", ".jpg", ".jpeg")):
        text = image_to_text(path)
    elif path.lower().endswith(".pdf"):
        text = pdf_to_text(path)
    else:
        text = open(path).read()
    build_faiss_index(text, sid)
    return {"status": "done"}

# 3) Query
@app.post("/query/{sid}")
def query(sid: str, payload: dict):
    question = payload.get("question")

    # Load FAISS index and chunks
    index, chunks = load_index_and_chunks(sid)

    # Convert chunks to LangChain Documents
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Build a FAISS-backed vectorstore using HuggingFaceEmbeddings
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, hf_embeddings)
    retriever = vectorstore.as_retriever()

    # Initialize the Gemini LLM via Google GenAI wrapper
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        api_key=os.getenv("GOOGLE_API_KEY"),
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Set up RetrievalQA chain with custom prompt
    qa = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    # Run the chain
    answer = qa.run(query=question)
    return {"answer": answer}

# 4) Status
@app.get("/status/{sid}")
def status(sid: str):
    idx_path = f"./data/{sid}/faiss.index"
    return {"processed": os.path.exists(idx_path)}

# 5) Cleanup
@app.delete("/cleanup/{sid}")
def cleanup(sid: str):
    session_dir = f"./data/{sid}"
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    return {"deleted": True}