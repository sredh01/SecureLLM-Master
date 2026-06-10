import ssl
import json
import os
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

WINDOW_NAME = "full"
WINDOW_START = None
WINDOW_END = None

OUTPUT_BASE_DIR = PROJECT_ROOT
DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, "data_full")

DIRS = {
    k: os.path.join(DATA_ROOT_DIR, k, "cisa_kev", WINDOW_NAME)
    for k in ("raw", "clean", "metadata", "manifests", "logs")
}
for folder in DIRS.values():
    os.makedirs(folder, exist_ok=True)

KEV_JSON_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
KEV_SCHEMA_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities_schema.json"


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


def save_json(path: str, payload: dict | list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def fetch_json_url(url: str) -> dict:
    ssl_context = ssl._create_unverified_context()
    with urlopen(url, context=ssl_context) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_kev_entries() -> list[dict]:
    payload = fetch_json_url(KEV_JSON_URL)
    vulns = payload.get("vulnerabilities", [])
    return vulns if isinstance(vulns, list) else []


def filter_rows_for_window(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if (row.get("cveID") or "").strip()
    ]


def build_metadata(row: dict) -> dict:
    cve_id = row.get("cveID", "").strip()
    vendor = row.get("vendorProject", "").strip()
    product = row.get("product", "").strip()
    vuln_name = row.get("vulnerabilityName", "").strip()
    short_description = row.get("shortDescription", "").strip()
    required_action = row.get("requiredAction", "").strip()
    due_date = row.get("dueDate", "").strip()
    notes = row.get("notes", "").strip()
    ransomware = row.get("knownRansomwareCampaignUse", "").strip()
    cwes_value = row.get("cwes", [])
    if isinstance(cwes_value, list):
        cwes = ", ".join(str(x).strip() for x in cwes_value if str(x).strip())
    else:
        cwes = str(cwes_value).strip()

    title_bits = [bit for bit in (vendor, product, vuln_name) if bit]
    title = " | ".join(title_bits) if title_bits else cve_id

    return {
        "source": KEV_JSON_URL,
        "scraped_utc": utc_now(),
        "window_name": WINDOW_NAME,
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "data_sources": ["cisa_kev"],
        "cve_id": cve_id,
        "title": title,
        "vendor_project": vendor,
        "product": product,
        "vulnerability_name": vuln_name,
        "description": short_description,
        "kev_date_added": row.get("dateAdded", "").strip(),
        "kev_due_date": due_date,
        "required_action": required_action,
        "notes": notes,
        "known_ransomware_campaign_use": ransomware,
        "cwes": cwes,
        "raw_row": row,
    }


def save_clean_text(cve_id: str, metadata: dict):
    lines = [
        f"SOURCE: {metadata.get('source', '')}",
        f"SCRAPED_UTC: {metadata.get('scraped_utc', '')}",
        f"WINDOW_NAME: {metadata.get('window_name', '')}",
        f"WINDOW_START: {metadata.get('window_start', '')}",
        f"WINDOW_END: {metadata.get('window_end', '')}",
        "",
        f"CVE ID: {metadata.get('cve_id', '')}",
        f"Title: {metadata.get('title', '')}",
        f"Vendor Project: {metadata.get('vendor_project', '')}",
        f"Product: {metadata.get('product', '')}",
        f"Vulnerability Name: {metadata.get('vulnerability_name', '')}",
        "",
        "Description:",
        metadata.get("description", ""),
        "",
        f"KEV Date Added: {metadata.get('kev_date_added', '')}",
        f"KEV Due Date: {metadata.get('kev_due_date', '')}",
        f"Required Action: {metadata.get('required_action', '')}",
        f"Known Ransomware Campaign Use: {metadata.get('known_ransomware_campaign_use', '')}",
        f"CWEs: {metadata.get('cwes', '')}",
        f"Notes: {metadata.get('notes', '')}",
        f"Data Sources: {', '.join(metadata.get('data_sources', []))}",
    ]
    with open(output_path(cve_id, "clean"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_run_manifest(cve_ids: list[str], run_id: str):
    path = os.path.join(DIRS["manifests"], f"run_manifest_{WINDOW_NAME}_{run_id}.txt")
    header = [
        f"WINDOW_NAME: {WINDOW_NAME}",
        f"WINDOW_START: {WINDOW_START}",
        f"WINDOW_END: {WINDOW_END}",
        f"KEV_JSON_URL: {KEV_JSON_URL}",
        f"KEV_SCHEMA_URL: {KEV_SCHEMA_URL}",
        f"ROWS_MATCHED: {len(cve_ids)}",
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
    print("WINDOW_START:", WINDOW_START)
    print("WINDOW_END:", WINDOW_END)
    print("OUTPUT_BASE_DIR:", OUTPUT_BASE_DIR)
    print("DATA_ROOT_DIR:", DATA_ROOT_DIR)
    print("KEV_JSON_URL:", KEV_JSON_URL)
    print("KEV_SCHEMA_URL:", KEV_SCHEMA_URL)
    run_id = timestamp_slug()
    print("RUN_ID:", run_id, "\n")

    try:
        rows = fetch_kev_entries()
        filtered_rows = filter_rows_for_window(rows)
    except Exception as e:
        print(f"[ERROR] Failed to fetch KEV full corpus :: {e}")
        return

    cve_ids = [(row.get("cveID") or "").strip() for row in filtered_rows if (row.get("cveID") or "").strip()]

    print(f"[INFO] Pulled {len(filtered_rows)} KEV rows in full corpus")
    print("\n".join(f"  -> {cve_id}" for cve_id in cve_ids[:20]))
    if len(cve_ids) > 20:
        print(f"  ... and {len(cve_ids) - 20} more")
    print("\n\n[ORDER CHECK]")
    print("\n".join(f"[{i}] {cid}" for i, cid in enumerate(cve_ids[:20])), end="\n\n")

    raw_dump_path = os.path.join(DIRS["raw"], f"kev_window_{WINDOW_NAME}_{run_id}.json")
    save_json(raw_dump_path, filtered_rows)

    manifest_path = write_run_manifest(cve_ids, run_id)
    print(f"[RAW WINDOW DUMP] {raw_dump_path}")
    print(f"[MANIFEST] {manifest_path}\n")

    saved = 0
    for row in filtered_rows:
        cve_id = (row.get("cveID") or "").strip()
        if not cve_id:
            continue
        print(f"[PROCESS] {cve_id}")
        if outputs_exist(cve_id):
            print(f"[SKIP] {cve_id} already has raw, metadata, and clean files")
            continue
        try:
            metadata = build_metadata(row)
            save_json(output_path(cve_id, "raw"), row)
            save_json(output_path(cve_id, "metadata"), metadata)
            save_clean_text(cve_id, metadata)
            saved += 1
            print(f"[SAVED] {cve_id}")
        except Exception as e:
            print(f"[ERROR] {cve_id} :: {e}")
            print(f"[FAILED LOG] {append_failed_cve(cve_id, str(e), run_id)}")

    print("\nDone.")
    print(f"KEV rows processed: {len(filtered_rows)}")
    print(f"KEV rows newly saved: {saved}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Failed log (if any): {os.path.join(DIRS['logs'], f'failed_cves_{WINDOW_NAME}_{run_id}.txt')}")


if __name__ == "__main__":
    from urllib.request import urlopen
    main()