# import streamlit as st
# import google.generativeai as genai
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# import os
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# # ===============================
# # Load API key
# # ===============================
# load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# # ===============================
# # Helper Functions
# # ===============================

# def extract_text_from_pdf(pdf_file):
#     """Extracts all text from uploaded PDF."""
#     pdf_reader = PdfReader(pdf_file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text() or ""
#     return text

# def split_text_into_chunks(text, chunk_size=500):
#     """Splits text into manageable chunks for embeddings."""
#     words = text.split()
#     return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# def create_embeddings(chunks):
#     """Creates embeddings for text chunks using SentenceTransformer."""
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(chunks)
#     return np.array(embeddings)

# def save_embeddings(embeddings, chunks):
#     """Stores embeddings in FAISS index."""
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)
#     return index, chunks

# def get_relevant_chunks(query, index, chunks, model, top_k=3):
#     """Retrieves top-k most relevant chunks for a query."""
#     query_emb = model.encode([query])
#     distances, indices = index.search(query_emb, top_k)
#     return [chunks[i] for i in indices[0]]

# # ===============================
# # Streamlit UI
# # ===============================

# st.set_page_config(page_title="RAG Tutor - Learn from Your PDF", layout="centered")

# st.title("📘 RAG Tutor - Learn from Your PDF (AI-powered)")

# uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])

# if uploaded_file:
#     with st.spinner("Processing your PDF... ⏳"):
#         # Step 1: Extract text
#         text = extract_text_from_pdf(uploaded_file)

#         # Step 2: Split into chunks
#         chunks = split_text_into_chunks(text)

#         # Step 3: Create embeddings
#         model = SentenceTransformer("all-MiniLM-L6-v2")
#         embeddings = model.encode(chunks)
#         index, chunks = save_embeddings(embeddings, chunks)

#     st.success("✅ PDF processed and embeddings created!")

#     # ===============================
#     # Ask questions
#     # ===============================
#     st.subheader("💬 Ask a question from your PDF:")
#     query = st.text_input("Type your question here:")

#     if query:
#         with st.spinner("Searching your PDF and generating answer... 🤔"):
#             # Step 4: Retrieve relevant chunks
#             relevant_chunks = get_relevant_chunks(query, index, chunks, model)

#             # Step 5: Create prompt
#             context = "\n\n".join(relevant_chunks)
#             prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

#             # Step 6: Use Gemini to generate response
#             try:
#                 # Change model name to compatible one
#                 ai_model = genai.GenerativeModel("gemini-2.5-flash")
 
#                 response = ai_model.generate_content(prompt)
#                 answer = response.text

#                 st.markdown("### 🧠 Answer:")
#                 st.write(answer)

#             except Exception as e:
#                 st.error(f"Error generating answer: {e}")


















































import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ===============================
# 🌍 Environment & API Setup
# ===============================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ===============================
# ⚙️ Helper Functions
# ===============================
def extract_text_from_pdf(pdf_file):
    """Extract all text from the uploaded PDF."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def split_text_into_chunks(text, chunk_size=500):
    """Split long text into smaller chunks for embeddings."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def create_embeddings(chunks):
    """Create embeddings using SentenceTransformer."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return np.array(embeddings), model

def save_embeddings(embeddings, chunks):
    """Store embeddings in FAISS index."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks

def get_relevant_chunks(query, index, chunks, model, top_k=3):
    """Retrieve top-k relevant chunks based on semantic similarity."""
    query_emb = model.encode([query])
    distances, indices = index.search(query_emb, top_k)
    return [chunks[i] for i in indices[0]]

# ===============================
# 🎨 Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="RAG Tutor - AI Learning Assistant",
    page_icon="📘",
    layout="centered"
)

# Custom CSS for cleaner UI
st.markdown("""
    <style>
    .stApp {
        background-color: #f8faff;
        font-family: "Segoe UI", sans-serif;
    }
    .title {
        font-size: 2.2rem;
        text-align: center;
        color: #0056b3;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #ffffff;
        border-left: 5px solid #007bff;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# 🧠 App Header
# ===============================
st.markdown('<div class="title">📘 RAG Tutor - Learn from Your PDF (AI-powered)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your PDF and ask anything you want to learn!</div>', unsafe_allow_html=True)

# ===============================
# 📤 PDF Upload Section
# ===============================
uploaded_file = st.file_uploader("📎 Upload a PDF file", type=["pdf"], help="Maximum 200MB")

if uploaded_file:
    with st.spinner("🔍 Extracting and embedding your PDF... please wait."):
        # Step 1: Extract text
        text = extract_text_from_pdf(uploaded_file)

        # Step 2: Split and embed
        chunks = split_text_into_chunks(text)
        embeddings, model = create_embeddings(chunks)
        index, chunks = save_embeddings(embeddings, chunks)

    st.success("✅ Your PDF has been processed successfully!")

    # ===============================
    # 💬 Q&A Section
    # ===============================
    st.markdown("---")
    st.subheader("💬 Ask a question from your PDF:")
    query = st.text_input("🧩 Type your question below:")

    if query:
        with st.spinner("🤔 Searching and generating your answer..."):
            # Step 3: Retrieve relevant chunks
            relevant_chunks = get_relevant_chunks(query, index, chunks, model)

            # Step 4: Prepare prompt for Gemini
            context = "\n\n".join(relevant_chunks)
            prompt = (
                f"Answer the question using only the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )

            try:
                # Use Gemini API
                ai_model = genai.GenerativeModel("gemini-2.5-flash")
                response = ai_model.generate_content(prompt)
                answer = response.text

                # Step 5: Display nicely
                st.markdown("### 🧠 Answer:")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Error generating answer: {e}")

else:
    st.info("👆 Please upload a PDF file to begin.")
