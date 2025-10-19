# embeddings_store.py
import numpy as np
import os

# Save embeddings and chunks to a file
def save_embeddings(chunks, embeddings, filename="embeddings_store.npz"):
    np.savez(filename, chunks=chunks, embeddings=embeddings)
    print(f"✅ Embeddings saved to {filename}")

# Load embeddings from file
def load_embeddings(filename="embeddings_store.npz"):
    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=True)
        chunks = data["chunks"]
        embeddings = data["embeddings"]
        print("✅ Embeddings loaded successfully!")
        return chunks, embeddings
    else:
        print("⚠️ No saved embeddings found. Please upload a PDF first.")
        return None, None
