import pickle
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "index" / "securellm_chunks.pkl"
INDEX_FILE = PROJECT_ROOT / "index" / "securellm_index.faiss"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def main():
    print("Loading chunks...")
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    texts = [item["text"] for item in chunks if item.get("text", "").strip()]
    print(f"Loaded {len(texts)} chunk texts")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Encoding chunk texts...")
    embeddings = embed_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dim}")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Saving FAISS index to: {INDEX_FILE}")
    faiss.write_index(index, str(INDEX_FILE))

    print("\nDone")
    print(f"Total indexed chunks: {index.ntotal}")
    print(f"Index saved to: {INDEX_FILE}")


if __name__ == "__main__":
    main()
