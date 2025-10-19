# utils.py
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# 1️⃣ Read text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# 2️⃣ Split text into chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# 3️⃣ Create embeddings (optional helper)
def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return np.array(embeddings)
