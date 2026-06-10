# SecureLLM: A Retrieval Augmented Framework for Cybersecurity Intelligence Analysis
## Overview
SecureLLM is a cybersecurity focused language model framework built using public vulnerability, exploit, malware, and threat intelligence sources. The project combines domain adaptation and retrieval augmented generation (RAG) to support cybersecurity question answering and intelligence analysis.

This repository contains the code used to reproduce the SecureLLM data ingestion, corpus construction, training, retrieval, and benchmarking pipeline. Public datasets must be obtained separately from their original sources.

The framework consists of:

* Data ingestion and normalization
* Unified cybersecurity corpus construction
* Domain adaptation of Qwen2.5 1.5B Instruct
* FAISS based retrieval augmentation
* Automated benchmark evaluation

## Repository Structure
```text
├── benchmark/
│   ├── Benchmark evaluation scripts
│
├── corpus/
│   ├── Benchmark prompt datasets
│
├── ingest/
│   ├── Data collection and preprocessing scripts
│
├── rag/
│   ├── Chunking, indexing, retrieval, and inference scripts
│
├── training/
│   ├── Domain adaptation training scripts
│
└── requirements.txt
```
Generated artifacts such as corpus files, FAISS indexes, benchmark outputs, logs, and model checkpoints are intentionally excluded from this repository and can be regenerated using the provided pipeline.

## Dataset Sources
The SecureLLM corpus was constructed from the following public cybersecurity resources:

* National Vulnerability Database (NVD)
  * https://nvd.nist.gov/developers/vulnerabilities

* CISA Known Exploited Vulnerabilities (KEV)
  * https://www.cisa.gov/known-exploited-vulnerabilities-catalog

* ExploitDB
  * https://gitlab.com/exploit-database/exploitdb

* MalwareBazaar
  * https://bazaar.abuse.ch/export/#csv

## Dataset Requirements

Some datasets used by SecureLLM are not redistributed through this repository.

Before running the ingestion pipeline:

* Download the ExploitDB repository and update the local paths in `exploitdb_crawler_r2.py`.
* Download the MalwareBazaar CSV export (`full.csv.zip`) and update the local path in `malwareb_batch.py`.

NVD and CISA KEV data are collected directly through their public APIs.

## Corpus Construction
### Generate normalized source records
```bash
python ingest/nvd_crawler_r2.py
python ingest/cisa_kev_ingest_r2.py
python ingest/exploitdb_crawler_r2.py
python ingest/malwareb_batch.py
```
Build the unified corpus
```bash
python ingest/securellm_v1_corpus.py
```
Output:
```text
corpus/securellm_v1_corpus.jsonl
```
## Retrieval Pipeline
Build document chunks
```bash
python rag/build_securellm_chunks_v1.py
```
Build the FAISS index
```bash
python rag/build_securellm_faiss_v1.py
```
Outputs
```text
index/securellm_chunks.pkl
index/securellm_index.faiss
```
## Model Training
Train the domain adapted model
```bash
python training/train_securellm_domain_v1.py
```
The model is based on
```text
Qwen/Qwen2.5-1.5B-Instruct
```
## Inference
Standard inference
```bash
python rag/inference_securellm_v1.py
```
RAG enabled inference
```bash
python rag/inference_securellm_rag_v1.py
```
## Benchmarking
Non RAG Benchmarking
```bash
python benchmark/benchmark_securellm_nonRag_v1.py
```
RAG Benchmark
```bash
python benchmark/benchmark_securellm_rag_v1.py
```
Prompt file:
```text
corpus/benchmark_100prompts.txt
```

## Model Availability

The SecureLLM model weights are not included in this repository. Update the model path variables in the benchmark and inference scripts to point to a local model directory or a separately hosted model release.

The repository contains the complete data processing, retrieval, training, and benchmarking pipeline used to reproduce the SecureLLM framework.

## Citation

If you use SecureLLM in academic work, please cite:

Redhouse, S. (2026).
SecureLLM: A Retrieval Augmented Framework for Cybersecurity Intelligence Analysis.
