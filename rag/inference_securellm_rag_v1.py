import pickle
import faiss
import numpy as np
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Update to your local SecureLLM model path
MODEL_PATH = "/path/to/securellm_domain_v1"
INDEX_FILE = PROJECT_ROOT / "index" / "securellm_index.faiss"
CHUNKS_FILE = PROJECT_ROOT / "index" / "securellm_chunks.pkl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
MAX_NEW_TOKENS = 200


def load_retrieval():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)

    print("Loading chunk metadata...")
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    return embed_model, index, chunks


def retrieve_context(embed_model, index, chunks, query, top_k=TOP_K):
    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    distances, indices = index.search(query_embedding, top_k)

    retrieved_chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    return retrieved_chunks, distances[0]


def build_rag_prompt(user_query, retrieved_chunks):
    context_parts = []

    for item in retrieved_chunks:
        source = item.get("source", "")
        record_id = item.get("record_id", "")
        title = item.get("title", "")
        text = item.get("text", "")

        context_parts.append(
            f"Source: {source}\n"
            f"Record ID: {record_id}\n"
            f"Title: {title}\n"
            f"Chunk Text:\n{text}"
        )

    context_text = "\n\n".join(context_parts)

    return f"""You are SecureLLM, a cybersecurity assistant.

Use only the context below to answer the question.
If the context does not support the answer, say you are unsure.

Context:
{context_text}

Question:
{user_query}

Answer:
"""


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Running on: {device}")

    embed_model, index, chunks = load_retrieval()

    while True:
        query = input("\nEnter question, or type exit: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        retrieved_chunks, distances = retrieve_context(embed_model, index, chunks, query)

        print("\nRetrieved Context Preview:")
        for i, item in enumerate(retrieved_chunks, start=1):
            print(f"\nRank {i}")
            print("Source:", item.get("source", ""))
            print("Record ID:", item.get("record_id", ""))
            print("Title:", item.get("title", ""))
            print("Distance:", float(distances[i - 1]))
            print("Text Preview:", item.get("text", "")[:300])

        rag_prompt = build_rag_prompt(query, retrieved_chunks)

        inputs = tokenizer(
            rag_prompt,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Answer:" in decoded:
            response = decoded.split("Answer:", 1)[1].strip()
        else:
            response = decoded.strip()

        for stop_token in [
            "Question:",
            "User:",
            "Assistant:",
            "### System:",
            "### User:",
        ]:
            if stop_token in response:
                response = response.split(stop_token)[0].strip()

        print("\nResponse:")
        print(response)


if __name__ == "__main__":
    main()
