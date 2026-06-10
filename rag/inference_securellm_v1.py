import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Update to your local SecureLLM model path
MODEL_PATH = "/path/to/securellm_domain_v1"
MAX_NEW_TOKENS = 200


def build_prompt(user_query: str) -> str:
    return f"""You are SecureLLM, a cybersecurity focused language model.

Answer the question clearly and directly.

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

    while True:
        query = input("\nEnter question, or type exit: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        prompt = build_prompt(query)

        inputs = tokenizer(
            prompt,
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

        print("\nResponse:")
        print(response)


if __name__ == "__main__":
    main()
