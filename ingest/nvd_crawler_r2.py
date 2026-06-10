import json
import os
import re
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

SOURCE_NAME = "nvd"
WINDOW_NAME = "recent_2024_2026"

OUTPUT_BASE_DIR = PROJECT_ROOT
DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, "data_full")

WINDOW_BLOCKS = [
    ("2024_block_1", "2024-01-01T00:00:00.000Z", "2024-04-29T23:59:59.000Z"),
    ("2025_block_1", "2025-01-01T00:00:00.000Z", "2025-04-30T23:59:59.000Z"),
    ("2026_block_1", "2026-01-01T00:00:00.000Z", "2026-03-23T23:59:59.000Z"),
]

DIRS = {
    k: os.path.join(DATA_ROOT_DIR, k, SOURCE_NAME, WINDOW_NAME)
    for k in ("raw", "clean", "metadata", "manifests", "logs")
}
for folder in DIRS.values():
    os.makedirs(folder, exist_ok=True)

load_dotenv()
NVD_API_KEY = os.getenv("NATIONAL_VULNERABILITY_DATABASE_API_KEY")

CVE_LIMIT = None
REQUEST_DELAY = 0.6
API_PAGE_SIZE = 200
NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_DETAIL_BASE = "https://nvd.nist.gov/vuln/detail"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    **({"apiKey": NVD_API_KEY} if NVD_API_KEY else {})
}


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def timestamp_slug() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", value)


def output_path(cve_id: str, kind: str) -> str:
    ext = "txt" if kind == "clean" else "json"
    return os.path.join(DIRS[kind], f"{safe_filename(cve_id)}.{ext}")


def outputs_exist(cve_id: str) -> bool:
    return all(os.path.exists(output_path(cve_id, kind)) for kind in ("raw", "metadata", "clean"))


def save_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_json_with_retries(url: str, params: dict, max_retries: int = 4) -> dict:
    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if response.status_code == 429:
            wait_time = 5 * (attempt + 1)
            print(f"[RATE LIMIT] request hit 429, waiting {wait_time}s...")
            time.sleep(wait_time)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Failed after retries for {url}")


def fetch_cve_ids_for_block(block_name: str, window_start: str, window_end: str, limit: int | None = None, max_retries: int = 4) -> list[str]:
    cve_ids = []
    seen = set()
    start_index = 0

    while limit is None or len(cve_ids) < limit:
        params = {
            "resultsPerPage": API_PAGE_SIZE,
            "startIndex": start_index,
            "pubStartDate": window_start,
            "pubEndDate": window_end,
        }
        data = get_json_with_retries(NVD_API_BASE, params, max_retries=max_retries)
        vulnerabilities = data.get("vulnerabilities", [])
        if not vulnerabilities:
            break

        for item in vulnerabilities:
            cve_id = item.get("cve", {}).get("id", "").strip()
            if cve_id and cve_id not in seen:
                seen.add(cve_id)
                cve_ids.append(cve_id)
                if limit is not None and len(cve_ids) >= limit:
                    return cve_ids

        start_index += API_PAGE_SIZE
        if start_index >= data.get("totalResults", 0):
            break

    return cve_ids


def extract_description(cve_obj: dict) -> str:
    descriptions = cve_obj.get("descriptions", [])
    english = next((d.get("value", "").strip() for d in descriptions if d.get("lang") == "en"), "")
    return english or (descriptions[0].get("value", "").strip() if descriptions else "")


def extract_title(cve_id: str, description: str) -> str:
    if not description:
        return cve_id
    first_sentence = description.split(".", 1)[0].strip()
    return f"{cve_id} {first_sentence[:117].rstrip() + '...' if len(first_sentence) > 120 else first_sentence}"


def extract_cvss_and_severity(cve_obj: dict) -> tuple[str, str]:
    metrics = cve_obj.get("metrics", {})
    ordered_items = []
    for key in ["cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
        ordered_items.extend(sorted(
            metrics.get(key, []),
            key=lambda item: (
                item.get("source") != "nvd@nist.gov",
                item.get("type") != "Primary",
            ),
        ))

    for item in ordered_items:
        cvss_data = item.get("cvssData", {})
        score = cvss_data.get("baseScore", "")
        severity = cvss_data.get("baseSeverity", "") or item.get("baseSeverity", "")
        if score != "" or severity != "":
            return (str(score) if score != "" else "", severity)
    return "", ""


def extract_cwe(cve_obj: dict) -> tuple[str, str]:
    for weakness in cve_obj.get("weaknesses", []):
        for desc in weakness.get("description", []):
            value = desc.get("value", "").strip()
            if desc.get("lang") == "en" and value.startswith("CWE-"):
                parts = value.split(" ", 1)
                return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
    return "", ""


def build_metadata(cve_id: str, api_payload: dict, block_name: str, window_start: str, window_end: str) -> dict:
    vulnerabilities = api_payload.get("vulnerabilities", [])
    if not vulnerabilities:
        raise ValueError(f"No vulnerability record returned for {cve_id}")

    cve_obj = vulnerabilities[0].get("cve", {})
    description = extract_description(cve_obj)
    cvss_score, severity = extract_cvss_and_severity(cve_obj)
    cwe_id, cwe_name = extract_cwe(cve_obj)
    references = [r.get("url", "").strip() for r in cve_obj.get("references", []) if r.get("url", "").strip()]

    return {
        "source": f"{NVD_DETAIL_BASE}/{cve_id}",
        "scraped_utc": utc_now(),
        "window_name": WINDOW_NAME,
        "window_start": window_start,
        "window_end": window_end,
        "window_block": block_name,
        "data_sources": ["nvd"],
        "cve_id": cve_id,
        "title": extract_title(cve_id, description),
        "description": description,
        "severity": severity,
        "cvss_score": cvss_score,
        "cwe_id": cwe_id,
        "cwe_name": cwe_name,
        "published_date": cve_obj.get("published", "")[:10],
        "updated_date": cve_obj.get("lastModified", "")[:10],
        "references": references,
        "exploit_available": False,
        "exploit_source": "",
        "exploit_url": "",
        "exploit_id": "",
        "malware_source": "",
        "malware_reference": "",
    }


def save_clean_text(cve_id: str, metadata: dict):
    top_fields = ["SOURCE", "SCRAPED_UTC", "WINDOW_NAME", "WINDOW_START", "WINDOW_END", "WINDOW_BLOCK"]
    body_fields = [
        "CVE ID", "Title", "Severity", "CVSS Score", "CWE ID", "CWE Name",
        "Published Date", "Updated Date", "Exploit Available", "Exploit Source",
        "Exploit URL", "Exploit ID", "Malware Source", "Malware Reference", "Data Sources",
    ]
    value_map = {
        "SOURCE": metadata.get("source", ""),
        "SCRAPED_UTC": metadata.get("scraped_utc", ""),
        "WINDOW_NAME": metadata.get("window_name", ""),
        "WINDOW_START": metadata.get("window_start", ""),
        "WINDOW_END": metadata.get("window_end", ""),
        "WINDOW_BLOCK": metadata.get("window_block", ""),
        "CVE ID": metadata.get("cve_id", ""),
        "Title": metadata.get("title", ""),
        "Severity": metadata.get("severity", ""),
        "CVSS Score": metadata.get("cvss_score", ""),
        "CWE ID": metadata.get("cwe_id", ""),
        "CWE Name": metadata.get("cwe_name", ""),
        "Published Date": metadata.get("published_date", ""),
        "Updated Date": metadata.get("updated_date", ""),
        "Exploit Available": metadata.get("exploit_available", False),
        "Exploit Source": metadata.get("exploit_source", ""),
        "Exploit URL": metadata.get("exploit_url", ""),
        "Exploit ID": metadata.get("exploit_id", ""),
        "Malware Source": metadata.get("malware_source", ""),
        "Malware Reference": metadata.get("malware_reference", ""),
        "Data Sources": ", ".join(metadata.get("data_sources", [])),
    }

    lines = [*(f"{k}: {value_map[k]}" for k in top_fields), "", f"CVE ID: {value_map['CVE ID']}", f"Title: {value_map['Title']}", "", "Description:", metadata.get("description", ""), ""]
    lines.extend(f"{k}: {value_map[k]}" for k in body_fields[2:])
    lines.extend(["", "References:", *metadata.get("references", [])])

    with open(output_path(cve_id, "clean"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_run_manifest(cve_ids: list[str], run_id: str):
    path = os.path.join(DIRS["manifests"], f"run_manifest_{WINDOW_NAME}_{run_id}.txt")
    header = [
        f"WINDOW_NAME: {WINDOW_NAME}",
        f"OUTPUT_BASE_DIR: {OUTPUT_BASE_DIR}",
        f"DATA_ROOT_DIR: {DATA_ROOT_DIR}",
        f"CVE_LIMIT: {CVE_LIMIT}",
        f"API_PAGE_SIZE: {API_PAGE_SIZE}",
        "WINDOW_BLOCKS:",
        *[f"{name} :: {start} :: {end}" for name, start, end in WINDOW_BLOCKS],
        "ORDER:",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + [f"[{i}] {cve_id}" for i, cve_id in enumerate(cve_ids)]))
    return path


def append_failed_cve(cve_id: str, error_message: str, run_id: str):
    path = os.path.join(DIRS["logs"], f"failed_cves_{WINDOW_NAME}_{run_id}.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{cve_id} :: {error_message}\n")
    return path


def main():
    print("WINDOW_NAME:", WINDOW_NAME)
    print("OUTPUT_BASE_DIR:", OUTPUT_BASE_DIR)
    print("DATA_ROOT_DIR:", DATA_ROOT_DIR)
    print("CVE_LIMIT:", CVE_LIMIT if CVE_LIMIT is not None else "ALL_IN_SCOPE")
    print("API_PAGE_SIZE:", API_PAGE_SIZE)
    print("NVD API key loaded:", bool(NVD_API_KEY))
    run_id = timestamp_slug()
    print("RUN_ID:", run_id, "\n")

    try:
        all_cve_ids = []
        seen = set()

        for block_name, block_start, block_end in WINDOW_BLOCKS:
            print(f"[FETCH BLOCK] {block_name} :: {block_start} -> {block_end}")
            block_ids = fetch_cve_ids_for_block(block_name, block_start, block_end, limit=None)
            print(f"[BLOCK COUNT] {block_name} -> {len(block_ids)} CVEs")

            for cve_id in block_ids:
                if cve_id not in seen:
                    seen.add(cve_id)
                    all_cve_ids.append((cve_id, block_name, block_start, block_end))

        if CVE_LIMIT is not None:
            all_cve_ids = all_cve_ids[:CVE_LIMIT]

        cve_ids_only = [item[0] for item in all_cve_ids]

    except Exception as e:
        print(f"[ERROR] Failed to fetch CVE list :: {e}")
        return

    print(f"[INFO] Pulled {len(cve_ids_only)} CVE IDs")
    print(f"[INFO] Using recent scope: {WINDOW_NAME}")
    print("\n".join(f"  -> {cve_id}" for cve_id in cve_ids_only[:20]))
    if len(cve_ids_only) > 20:
        print(f"  ... and {len(cve_ids_only) - 20} more")
    print("\n\n[ORDER CHECK]")
    print("\n".join(f"[{i}] {cid}" for i, cid in enumerate(cve_ids_only[:20])), end="\n\n")

    manifest_path = write_run_manifest(cve_ids_only, run_id)
    print(f"[MANIFEST] {manifest_path}\n")

    saved = 0
    for cve_id, block_name, block_start, block_end in all_cve_ids:
        print(f"[FETCH] {cve_id}")
        if outputs_exist(cve_id):
            print(f"[SKIP] {cve_id} already has raw, metadata, and clean files")
            continue
        try:
            payload = get_json_with_retries(NVD_API_BASE, {"cveId": cve_id})
            metadata = build_metadata(cve_id, payload, block_name, block_start, block_end)
            save_json(output_path(cve_id, "raw"), payload)
            save_json(output_path(cve_id, "metadata"), metadata)
            save_clean_text(cve_id, metadata)
            saved += 1
            print(f"[SAVED] {cve_id}")
        except Exception as e:
            print(f"[ERROR] {cve_id} :: {e}")
            print(f"[FAILED LOG] {append_failed_cve(cve_id, str(e), run_id)}")
        time.sleep(REQUEST_DELAY)

    print("\nDone.")
    print(f"CVEs processed: {len(cve_ids_only)}")
    print(f"CVEs newly saved: {saved}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Failed log (if any): {os.path.join(DIRS['logs'], f'failed_cves_{WINDOW_NAME}_{run_id}.txt')}")


if __name__ == "__main__":
    main()