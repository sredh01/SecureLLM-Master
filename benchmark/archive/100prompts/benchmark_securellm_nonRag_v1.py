import csv
import time
import torch

from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================================
# CONFIG
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = PROJECT_ROOT / "corpus" / "benchmark_100prompts.txt"

# Update to your local SecureLLM model path
SECURELLM_MODEL = "/path/to/securellm_domain"
QWEN_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

CHECKPOINT_FILE = (PROJECT_ROOT / "benchmark" / "securellm_benchmark_checkpoint.csv")

MAX_NEW_TOKENS = 200

MODELS = [
    "qwen_base",
    "securellm_domain_v1",
]

# ==========================================================
# DEVICE
# ==========================================================

def get_device():
    if torch.cuda.is_available():
        print("GPU detected:", torch.cuda.get_device_name(0))
        return "cuda"
    else:
        print("No GPU detected. Running in CPU mode.")
        return "cpu"

# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(model_name):
    device = get_device()

    model_map = {
        "qwen_base": QWEN_BASE_MODEL,
        "securellm_domain_v1": SECURELLM_MODEL,
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


def build_prompt(user_prompt: str) -> str:
    return f"""You are SecureLLM, a cybersecurity focused language model.

Answer the question clearly and directly.
Do not invent facts.
If you are unsure, say that you are unsure.

Question:
{user_prompt}

Answer:
"""

# ==========================================================
# GENERATE
# ==========================================================

def generate_response(model, tokenizer, prompt):
    full_prompt = build_prompt(prompt)

    model_device = next(model.parameters()).device

    inputs = tokenizer(
        full_prompt,
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
# LIGHTWEIGHT AUTO SIGNALS
# ==========================================================

CYBER_TERMS = [
    "cve", "cvss", "cwe", "exploit", "vulnerability", "malware",
    "hash", "ransomware", "privilege", "authentication", "patch",
    "threat", "ioc", "command and control", "remote code execution"
]

UNCERTAINTY_TERMS = [
    "not sure", "unsure", "uncertain", "may", "might", "possibly"
]


def evaluate_lightweight(prompt, response):
    response_lower = response.lower()

    cyber_term_hits = sum(1 for term in CYBER_TERMS if term in response_lower)
    uncertainty_hits = sum(1 for term in UNCERTAINTY_TERMS if term in response_lower)

    return {
        "response_length": len(response),
        "word_count": len(response.split()),
        "cyber_term_hits": cyber_term_hits,
        "uncertainty_hits": uncertainty_hits,
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

            response, latency = generate_response(
                model,
                tokenizer,
                prompt
            )

            auto_eval = evaluate_lightweight(prompt, response)

            row = {
                "prompt_id": prompt_id,
                "model": model_name,
                "prompt_type": prompt_type,
                "prompt": prompt,
                "response": response,
                "latency_seconds": round(latency, 4),
                **auto_eval,
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
    output_file = (PROJECT_ROOT / "benchmark" / f"securellm_benchmark_{timestamp}.csv")
    save_results(results, output_file)

    print("\nBenchmark Complete.")
    print("Saved to:", output_file)
    print("Checkpoint file:", CHECKPOINT_FILE)

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    print("Starting SecureLLM Benchmark...")
    run_benchmark()
