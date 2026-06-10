import csv
import pickle
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================================
# CONFIG
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = PROJECT_ROOT / "corpus" / "benchmark_100prompts.txt"

# Update to your local SecureLLM model path
SECURELLM_MODEL = "/path/to/securellm_domain"
QWEN_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

INDEX_FILE = PROJECT_ROOT / "index" / "securellm_index.faiss"
CHUNKS_FILE = PROJECT_ROOT / "index" / "securellm_chunks.pkl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
MAX_NEW_TOKENS = 200

MODELS = [
    "qwen_base",
    "securellm_domain",
]

CHECKPOINT_FILE = (
    PROJECT_ROOT / "benchmark" / "securellm_rag_benchmark_checkpoint.csv"
)

REFUSAL_PHRASES = [
    "i am unsure",
    "i'm unsure",
    "not enough context",
    "context does not support",
    "the context does not support",
    "i do not know",
    "i don't know",
    "not supported by the context",
]

# ==========================================================
# DEVICE
# ==========================================================

def get_device():
    if torch.cuda.is_available():
        print("GPU detected:", torch.cuda.get_device_name(0))
        return "cuda"
    print("No GPU detected. Running in CPU mode.")
    return "cpu"

# ==========================================================
# LOAD RETRIEVAL
# ==========================================================

def load_retrieval():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_FILE))

    print("Loading chunk metadata...")
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    print(f"Retrieval ready. Total chunks: {len(chunks)}")
    return embed_model, index, chunks


def retrieve_context(embed_model, index, chunks, query, top_k=TOP_K):
    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    distances, indices = index.search(query_embedding, top_k)

    retrieved_chunks = []
    retrieved_sources = []

    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            item = chunks[idx]
            retrieved_chunks.append(item["text"])
            retrieved_sources.append(
                f"{item.get('source', '')} | {item.get('record_id', '')} | chunk {item.get('chunk_id', '')}"
            )

    return retrieved_chunks, retrieved_sources, distances[0]

# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(model_name):
    device = get_device()

    model_map = {
        "qwen_base": QWEN_BASE_MODEL,
        "securellm_domain": SECURELLM_MODEL,
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = model_map[model_name]

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    if device == "cuda":
        print(f"Loading model for {model_name} in GPU mode...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print(f"Loading model for {model_name} in CPU mode...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.to("cpu")

    model.eval()
    print(f"Model ready: {model_name}")
    return model, tokenizer

# ==========================================================
# PROMPT HELPERS
# ==========================================================

def infer_prompt_type(prompt: str) -> str:
    lower = prompt.lower().strip()

    if "compare" in lower or "difference" in lower or "distinction" in lower:
        return "comparison"
    if "why" in lower or "reason" in lower or "logic" in lower:
        return "reasoning"
    if "suppose" in lower or "imagine" in lower or "if a system" in lower or "what kind of damage" in lower:
        return "scenario"
    if "classify" in lower or "identify which one" in lower or "place it" in lower or "label it" in lower:
        return "classification"
    if "explain" in lower or "walk me through" in lower or "spell out" in lower:
        return "explanation"
    return "concept"


def build_rag_prompt(user_prompt: str, context_chunks: list[str]) -> str:
    context_text = "\n\n".join(context_chunks)

    return f"""You are SecureLLM, a cybersecurity assistant.

Use only the context below to answer the question.
If the context does not support the answer, say you are unsure.

Context:
{context_text}

Question:
{user_prompt}

Answer:
"""

# ==========================================================
# GENERATE
# ==========================================================

def generate_response(model, tokenizer, prompt):
    model_device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    latency = time.time() - start

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

    return response, latency

# ==========================================================
# EVALUATE
# ==========================================================

def contains_refusal_or_uncertainty(response: str) -> int:
    lower = response.lower()
    return int(any(p in lower for p in REFUSAL_PHRASES))


def evaluate(response, context_chunks, distances, embed_model):
    response_embedding = embed_model.encode(
        [response],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    context_embedding = embed_model.encode(
        [" ".join(context_chunks)],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarity = cosine_similarity(response_embedding, context_embedding)[0][0]
    contains_refusal = contains_refusal_or_uncertainty(response)

    hallucination_risk = int(similarity < 0.5 and contains_refusal == 0)

    score = 0
    if similarity > 0.65:
        score += 3
    if hallucination_risk == 0:
        score += 2
    if contains_refusal:
        score += 1

    return {
        "grounded_similarity": float(similarity),
        "hallucination_risk": hallucination_risk,
        "contains_refusal": contains_refusal,
        "retrieval_avg_distance": float(np.mean(distances)),
        "response_length": len(response),
        "word_count": len(response.split()),
        "final_score": score,
    }

# ==========================================================
# SAVE
# ==========================================================

def save_results(results, filename):
    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
            escapechar="\\"
        )
        writer.writeheader()
        writer.writerows(results)

# ==========================================================
# BENCHMARK
# ==========================================================

def run_benchmark(prompt_file=PROMPT_FILE):
    embed_model, index, chunks = load_retrieval()

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [p.strip() for p in f if p.strip()]

    print(f"Loaded {len(prompts)} prompts from {prompt_file}")

    results = []

    for model_idx, model_name in enumerate(MODELS, start=1):
        print("\n==================================================")
        print(f"Starting model {model_idx}/{len(MODELS)}: {model_name}")
        print("==================================================")

        model, tokenizer = load_model(model_name)

        for prompt_id, prompt in enumerate(prompts, start=1):
            prompt_type = infer_prompt_type(prompt)

            context_chunks, retrieved_sources, distances = retrieve_context(
                embed_model,
                index,
                chunks,
                prompt
            )

            rag_prompt = build_rag_prompt(prompt, context_chunks)
            response, latency = generate_response(model, tokenizer, rag_prompt)
            evaluation = evaluate(response, context_chunks, distances, embed_model)

            row = {
                "prompt_id": prompt_id,
                "model": model_name,
                "prompt_type": prompt_type,
                "prompt": prompt,
                "response": response,
                "sources": " || ".join(retrieved_sources),
                "latency_seconds": round(latency, 4),
                **evaluation,
                "manual_domain_relevance": "",
                "manual_technical_correctness": "",
                "manual_factuality": "",
                "manual_clarity": "",
                "manual_hallucination_risk": "",
                "manual_final_score": "",
                "manual_notes": "",
            }

            results.append(row)

            if len(results) % 10 == 0:
                print(f"Saving checkpoint at {len(results)} total rows...")
                save_results(results, CHECKPOINT_FILE)

        print(f"Finished model {model_idx}/{len(MODELS)}: {model_name}")

        save_results(results, CHECKPOINT_FILE)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        PROJECT_ROOT / "benchmark" / f"securellm_rag_benchmark_{timestamp}.csv"
    )
    save_results(results, output_file)

    print("\nBenchmark Complete.")
    print("Saved to:", output_file)
    print("Checkpoint file:", CHECKPOINT_FILE)


if __name__ == "__main__":
    print("Starting SecureLLM RAG Benchmark...")
    run_benchmark()
