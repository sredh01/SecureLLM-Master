import json
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "corpus" / "securellm_v1_corpus.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "index" / "securellm_chunks.pkl"

CHUNK_WORDS = 300
OVERLAP_WORDS = 60


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = OVERLAP_WORDS):
    words = text.split()

    if not words:
        return []

    chunks = []
    step = max(1, chunk_words - overlap_words)

    for start in range(0, len(words), step):
        end = start + chunk_words
        chunk_words_list = words[start:end]

        if not chunk_words_list:
            continue

        chunk = " ".join(chunk_words_list).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


def main():
    all_chunks = []
    total_records = 0
    total_chunks = 0

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            total_records += 1

            record_text = row.get("text", "").strip()
            if not record_text:
                continue

            record_chunks = chunk_text(record_text)

            for chunk_id, chunk_text_value in enumerate(record_chunks):
                chunk_record = {
                    "chunk_global_id": total_chunks,
                    "chunk_id": chunk_id,
                    "source": row.get("source", ""),
                    "record_id": row.get("record_id", ""),
                    "title": row.get("title", ""),
                    "document_type": row.get("document_type", ""),
                    "published_date": row.get("published_date", ""),
                    "updated_date": row.get("updated_date", ""),
                    "text": chunk_text_value,
                }
                all_chunks.append(chunk_record)
                total_chunks += 1

            if total_records % 5000 == 0:
                print(f"Processed {total_records} records, total chunks so far: {total_chunks}")

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print("\nDone")
    print(f"Records processed: {total_records}")
    print(f"Chunks created: {total_chunks}")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
