import os
import json
from pathlib import Path
from itertools import islice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, "data_full")
CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus")

SOURCE_DIRS = {
    "nvd": [
        {
            "clean": os.path.join(DATA_ROOT_DIR, "clean", "nvd"),
            "meta": os.path.join(DATA_ROOT_DIR, "metadata", "nvd"),
        }
    ],
    "exploitdb": [
        {
            "clean": os.path.join(DATA_ROOT_DIR, "clean", "exploitdb"),
            "meta": os.path.join(DATA_ROOT_DIR, "metadata", "exploitdb"),
        }
    ],
    "cisa_kev": [
        {
            "clean": os.path.join(DATA_ROOT_DIR, "clean", "cisa_kev"),
            "meta": os.path.join(DATA_ROOT_DIR, "metadata", "cisa_kev"),
        }
    ],
    "malwarebazaar": [
        {
            "clean": os.path.join(DATA_ROOT_DIR, "clean", "malwarebazaar"),
            "meta": os.path.join(DATA_ROOT_DIR, "metadata", "malwarebazaar"),
        }
    ],
}

TEST_SOURCE_DIRS = SOURCE_DIRS

# Change to true for testing and set # of max files. For full corpus change to False, and set max source to None.
USE_TEST_MODE = False
TEST_MAX_FILES_PER_SOURCE = None
TEST_OUTPUT_FILE = os.path.join(CORPUS_DIR, "securellm_v1_corpus_test.jsonl")

OUTPUT_FILE = os.path.join(CORPUS_DIR, "securellm_v1_corpus.jsonl")

def get_active_sources():
    return TEST_SOURCE_DIRS if USE_TEST_MODE else SOURCE_DIRS


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def read_json(path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def normalize_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def extract_structured_fields(source, meta, record_id):
    base = {
        "source": source,
        "record_id": record_id,
        "title": None,
        "description": None,
        "published_date": None,
        "updated_date": None,
        "tags": [],
        "references": [],
    }

    if source == "nvd":
        base.update({
            "cve_id": first_non_empty(meta.get("cve_id"), record_id),
            "title": first_non_empty(meta.get("title")),
            "description": first_non_empty(meta.get("description")),
            "severity": first_non_empty(meta.get("severity")),
            "cvss_score": first_non_empty(meta.get("cvss_score"), meta.get("cvss")),
            "cwe_id": first_non_empty(meta.get("cwe_id"), meta.get("cwe")),
            "cwe_name": first_non_empty(meta.get("cwe_name")),
            "published_date": first_non_empty(meta.get("published_date"), meta.get("published")),
            "updated_date": first_non_empty(meta.get("updated_date")),
            "exploit_available": meta.get("exploit_available"),
            "exploit_source": first_non_empty(meta.get("exploit_source")),
            "exploit_url": first_non_empty(meta.get("exploit_url")),
            "exploit_id": first_non_empty(meta.get("exploit_id")),
            "references": normalize_list(meta.get("references")),
            "tags": [t for t in [first_non_empty(meta.get("severity")), first_non_empty(meta.get("cwe_id"), meta.get("cwe"))] if t],
        })

    elif source == "cisa_kev":
        base.update({
            "cve_id": first_non_empty(meta.get("cve_id"), record_id),
            "title": first_non_empty(meta.get("title"), meta.get("vulnerability_name")),
            "description": first_non_empty(meta.get("description")),
            "published_date": first_non_empty(meta.get("kev_date_added"), meta.get("published_date")),
            "updated_date": first_non_empty(meta.get("kev_due_date"), meta.get("updated_date")),
            "vendor_project": first_non_empty(meta.get("vendor_project")),
            "product": first_non_empty(meta.get("product")),
            "vulnerability_name": first_non_empty(meta.get("vulnerability_name")),
            "kev_date_added": first_non_empty(meta.get("kev_date_added")),
            "kev_due_date": first_non_empty(meta.get("kev_due_date"), meta.get("due_date")),
            "required_action": first_non_empty(meta.get("required_action")),
            "known_ransomware_campaign_use": first_non_empty(meta.get("known_ransomware_campaign_use")),
            "cwes": first_non_empty(meta.get("cwes")),
            "notes": first_non_empty(meta.get("notes")),
            "references": normalize_list(meta.get("references")),
            "tags": [
                t for t in [
                    first_non_empty(meta.get("vendor_project")),
                    first_non_empty(meta.get("product")),
                    first_non_empty(meta.get("known_ransomware_campaign_use")),
                ] if t
            ],
        })

    elif source == "exploitdb":
        base.update({
            "record_key": first_non_empty(meta.get("record_key"), record_id),
            "exploit_id": first_non_empty(meta.get("exploit_id")),
            "title": first_non_empty(meta.get("exploit_title"), meta.get("title"), meta.get("description")),
            "description": first_non_empty(meta.get("description")),
            "published_date": first_non_empty(meta.get("date_published"), meta.get("date")),
            "updated_date": None,
            "author": first_non_empty(meta.get("author")),
            "platform": first_non_empty(meta.get("platform")),
            "type": first_non_empty(meta.get("type")),
            "port": first_non_empty(meta.get("port")),
            "file": first_non_empty(meta.get("file")),
            "exploit_url": first_non_empty(meta.get("exploit_url")),
            "exploit_available": meta.get("exploit_available"),
            "references": normalize_list(meta.get("references")),
            "tags": [
                t for t in [
                    first_non_empty(meta.get("platform")),
                    first_non_empty(meta.get("type")),
                    first_non_empty(meta.get("author")),
                ] if t
            ],
        })

    elif source == "malwarebazaar":
        mz = meta.get("malwarebazaar", {}) if isinstance(meta.get("malwarebazaar"), dict) else {}
        base.update({
            "external_id": first_non_empty(meta.get("external_id"), mz.get("sha256_hash")),
            "document_type": first_non_empty(meta.get("document_type")),
            "family_or_signature": first_non_empty(meta.get("family_or_signature"), mz.get("signature"), mz.get("family")),
            "title": first_non_empty(meta.get("title")),
            "summary": first_non_empty(meta.get("summary")),
            "description": first_non_empty(meta.get("description"), meta.get("summary")),
            "published_date": first_non_empty(meta.get("published_date"), mz.get("first_seen")),
            "updated_date": first_non_empty(meta.get("updated_date"), mz.get("last_seen"), mz.get("first_seen")),
            "record_datetime": first_non_empty(meta.get("record_datetime")),
            "file_name": first_non_empty(mz.get("file_name")),
            "file_type": first_non_empty(meta.get("file_type"), mz.get("file_type")),
            "file_type_mime": first_non_empty(mz.get("file_type_mime")),
            "reporter": first_non_empty(mz.get("reporter")),
            "sha256": first_non_empty(mz.get("sha256_hash"), meta.get("external_id")),
            "sha1": first_non_empty(mz.get("sha1_hash")),
            "md5": first_non_empty(mz.get("md5_hash")),
            "imphash": first_non_empty(mz.get("imphash")),
            "ssdeep": first_non_empty(mz.get("ssdeep")),
            "tlsh": first_non_empty(mz.get("tlsh")),
            "malware_reference": first_non_empty(meta.get("malware_reference"), meta.get("external_id")),
            "references": normalize_list(first_non_empty(meta.get("references"), mz.get("references"))),
            "tags": normalize_list(first_non_empty(meta.get("tags"), mz.get("tags"))),
        })

    return base


def pick_fields(source, structured):
    fields = []

    fields.append(f"Source: {source}")
    fields.append(f"Record ID: {structured['record_id']}")

    ordered_keys = [
        ("cve_id", "CVE"),
        ("title", "Title"),
        ("severity", "Severity"),
        ("cvss_score", "CVSS"),
        ("cwe_id", "CWE"),
        ("published_date", "Published"),
        ("updated_date", "Updated"),
        ("vendor_project", "Vendor"),
        ("product", "Product"),
        ("vulnerability_name", "Vulnerability Name"),
        ("kev_date_added", "KEV Date Added"),
        ("kev_due_date", "KEV Due Date"),
        ("required_action", "Required Action"),
        ("known_ransomware_campaign_use", "Known Ransomware Use"),
        ("record_key", "Record Key"),
        ("exploit_id", "Exploit ID"),
        ("author", "Author"),
        ("platform", "Platform"),
        ("type", "Type"),
        ("port", "Port"),
        ("exploit_url", "Exploit URL"),
        ("external_id", "External ID"),
        ("document_type", "Document Type"),
        ("family_or_signature", "Family"),
        ("file_name", "File Name"),
        ("file_type", "File Type"),
        ("file_type_mime", "MIME Type"),
        ("reporter", "Reporter"),
        ("sha256", "SHA256"),
        ("sha1", "SHA1"),
        ("md5", "MD5"),
        ("imphash", "IMPHASH"),
        ("ssdeep", "SSDEEP"),
        ("tlsh", "TLSH"),
        ("record_datetime", "Observed Datetime"),
        ("malware_reference", "Malware Reference"),
    ]

    for key, label in ordered_keys:
        value = structured.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        fields.append(f"{label}: {value}")

    tags = structured.get("tags") or []
    if tags:
        fields.append(f"Tags: {', '.join(str(tag) for tag in tags)}")

    description = structured.get("description")
    if description:
        fields.append("")
        fields.append("Description:")
        fields.append(description)

    references = structured.get("references") or []
    if references:
        fields.append("")
        fields.append("References:")
        fields.extend(str(ref) for ref in references)

    return "\n".join(fields)


def build_record(source, txt_path, meta_path):
    text_body = read_text(txt_path)
    if not text_body:
        return None, "empty"

    word_count = len(text_body.split())
    min_words_by_source = {
        "nvd": 20,
        "exploitdb": 8,
        "cisa_kev": 8,
        "malwarebazaar": 8,
    }
    min_words = min_words_by_source.get(source, 20)

    if word_count < min_words:
        return None, f"too_short_{word_count}"

    meta = read_json(meta_path)
    record_id = txt_path.stem
    structured = extract_structured_fields(source, meta, record_id)
    header = pick_fields(source, structured)
    full_text = f"{header}\n\nContent:\n{text_body}"

    record = dict(structured)
    record["text"] = full_text

    return record, None


def main():
    output_path = Path(TEST_OUTPUT_FILE if USE_TEST_MODE else OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_seen = 0
    total_kept = 0
    total_missing_meta = 0
    total_empty = 0
    total_too_short = 0

    with open(output_path, "w", encoding="utf-8") as out:
        active_sources = get_active_sources()
        print(f"Running in {'TEST' if USE_TEST_MODE else 'FULL'} mode")

        for source, path_groups in active_sources.items():
            source_seen = 0
            source_kept = 0
            source_empty = 0
            source_too_short = 0
            source_examples = []

            for paths in path_groups:
                clean_root = Path(paths["clean"])
                meta_root = Path(paths["meta"])

                if not clean_root.exists():
                    print(f"Missing clean directory for {source}: {clean_root}")
                    continue

                txt_files = sorted(
                    p for p in clean_root.rglob("*.txt")
                    if not p.name.startswith("._")
                )
                total_found = len(txt_files)

                if USE_TEST_MODE:
                    if TEST_MAX_FILES_PER_SOURCE is None:
                        txt_files = list(txt_files)
                        print(
                            f"{source}, root {clean_root}, found {total_found} txt files, "
                            f"using all {len(txt_files)} for testing"
                        )
                    else:
                        remaining = max(TEST_MAX_FILES_PER_SOURCE - source_seen, 0)
                        txt_files = list(islice(txt_files, remaining))
                        print(
                            f"{source}, root {clean_root}, found {total_found} txt files, "
                            f"using {len(txt_files)} for testing"
                        )
                else:
                    print(f"{source}, root {clean_root}, found {total_found} txt files, using all {total_found} for full build")

                for txt_path in txt_files:
                    total_seen += 1
                    source_seen += 1
                    rel = txt_path.relative_to(clean_root)
                    meta_path = meta_root / rel.with_suffix(".json")

                    if meta_path.name.startswith("._"):
                        continue

                    if not meta_path.exists():
                        total_missing_meta += 1

                    record, skip_reason = build_record(source, txt_path, meta_path)
                    if record is None:
                        if skip_reason == "empty":
                            total_empty += 1
                            source_empty += 1
                        elif skip_reason and skip_reason.startswith("too_short"):
                            total_too_short += 1
                            source_too_short += 1
                            if len(source_examples) < 3:
                                preview = read_text(txt_path)[:160].replace("\n", " ")
                                source_examples.append(
                                    f"{txt_path.name} ({skip_reason}) -> {preview}"
                                )
                        continue

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_kept += 1
                    source_kept += 1

                if USE_TEST_MODE and TEST_MAX_FILES_PER_SOURCE is not None and source_seen >= TEST_MAX_FILES_PER_SOURCE:
                    break

            print(
                f"{source}, kept {source_kept} records, "
                f"empty: {source_empty}, too short: {source_too_short}"
            )
            if source_examples:
                print(f"{source}, example skipped files:")
                for example in source_examples:
                    print(f"  {example}")

    print("\nDone")
    print(f"Total seen: {total_seen}")
    print(f"Total kept: {total_kept}")
    print(f"Missing metadata files: {total_missing_meta}")
    print(f"Empty files skipped: {total_empty}")
    print(f"Too short files skipped: {total_too_short}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()