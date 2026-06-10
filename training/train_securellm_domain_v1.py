import json
import logging
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_FILE = "/home/sredhouse/securellm_cluster/data/securellm_v1_corpus.jsonl"
OUTPUT_DIR = "/home/sredhouse/securellm_cluster/output/securellm_domain_v1"

MAX_LENGTH = 1024
VAL_RATIO = 0.02
SEED = 42


def get_device():
    if torch.cuda.is_available():
        logging.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    logging.info("No GPU detected, running in CPU mode.")
    return "cpu"


class SecureLLMTextDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        text = self.records[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        return item


def load_texts(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if text and text.strip():
                texts.append(text)
    return texts


def main():
    device = get_device()
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    logging.info("Loading corpus from JSONL...")
    texts = load_texts(DATA_FILE)
    logging.info(f"Total loaded text records: {len(texts)}")

    random.shuffle(texts)

    split_idx = int(len(texts) * (1 - VAL_RATIO))
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]

    logging.info(f"Train size: {len(train_texts)} | Eval size: {len(eval_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = SecureLLMTextDataset(train_texts, tokenizer, MAX_LENGTH)
    eval_dataset = SecureLLMTextDataset(eval_texts, tokenizer, MAX_LENGTH)

    logging.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        save_total_limit=2,
        fp16=False,
        bf16=use_bf16,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.0
            )
        ],
    )

    logging.info("Starting domain training...")
    trainer.train()

    logging.info("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logging.info("Done.")


if __name__ == "__main__":
    main()
