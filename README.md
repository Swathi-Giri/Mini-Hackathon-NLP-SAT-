# 🤖 Multi-Agent RAG Pipeline for Policy Analysis

> A modular, production-grade **Retrieval-Augmented Generation (RAG)** system built with a multi-agent architecture for deep analysis of international policy documents (OECD, IMF, UN). Combines classical NLP with modern transformer models and hybrid vector search.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Breakdown](#agent-breakdown)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Results & Outputs](#results--outputs)
- [Sample Queries](#sample-queries)

---

## Overview

This pipeline ingests complex policy PDFs, processes them through a series of specialized agents, and answers multi-hop, cross-document questions with source citations, factuality checks, and structured debate. It was built as part of a Mini Hackathon NLP project.

**Key capabilities:**
- 📄 Ingest and chunk PDFs with page-level citations
- 🔍 Hybrid retrieval: BM25 + Pinecone dense vectors with Cross-Encoder reranking
- 🧠 LDA topic modeling across 11 policy documents
- 🗣️ Structured debate (Pro vs. Con) on policy questions
- ✅ Factuality verification via NLI (BART-large-MNLI)
- 🛡️ PII redaction and prompt injection filtering
- 📊 Performance visualization and auto-tuning

---

## Architecture

The pipeline is a directed graph of agents. Each agent has a single responsibility and passes structured output to the next.

```
PDFs
 │
 ▼
[PDFIngestionAgent] ──► text chunks + page metadata
 │
 ▼
[PreprocessorAgent] ──► tokenized, lemmatized, NER-tagged chunks
 │
 ▼
[TopicModelAgent] ──► 10 LDA topics
 │
 ▼
[EmbeddingAgent] ──► dense vectors (SentenceTransformers)
 │
 ├──► [RetrieverAgent]           (Baseline: BM25 + Pinecone)
 └──► [RetrieverExperimentAgent] (Advanced: + Cross-Encoder Reranking)
         │
         ▼
     [PlannerAgent] ──► query decomposition + language detection
         │
         ▼
     [SummarizerAgent] ──► structured summary with citations [src: file p.X]
         │
         ▼
     [DebateAgents A & B] ──► Pro/Con arguments on policy questions
         │
         ▼
     [VerifierAgent] ──► NLI factuality + cosine semantic alignment score
         │
         ▼
     [GuardrailsAgent] ──► PII redaction + prompt injection filter
         │
         ▼
     [MemoryAgent] ──► log α, k, latency, confidence
         │
         ▼
     [VisualizerAgent] ──► plots + agent graph
```

---

## Agent Breakdown

| Agent | File | Function | Techniques |
|---|---|---|---|
| **PDFIngestionAgent** | `task_1_ingestion.py` | Parse PDFs, extract text & tables, chunk semantically | `pdfplumber`, `RecursiveCharacterTextSplitter` |
| **PreprocessorAgent** | `task_2_modeling.py` | Tokenize, lemmatize, NER tagging | `spaCy (en_core_web_sm)` |
| **TopicModelAgent** | `task_2_modeling.py` | Discover 10 topics using LDA | `scikit-learn` |
| **EmbeddingAgent** | `task_2_modeling.py` | Build dense embeddings for chunks | `sentence-transformers` |
| **RetrieverAgent** | `retriever_agent.py` | Baseline Hybrid RAG (BM25 + Pinecone) | `rank-bm25`, `pinecone-client` |
| **RetrieverExperimentAgent** | `retriever_experiment_agent.py` | Advanced RAG with Cross-Encoder Reranking | `sentence-transformers CrossEncoder` |
| **PlannerAgent** | `planner_agent.py` | Decompose complex queries & detect language (EN/DE) | `langdetect`, `google/flan-t5-large` |
| **SummarizerAgent** | `summarizer_agent.py` | Structured summary with inline citations | `google/flan-t5-large` |
| **DebateAgents A & B** | `task_5_debate.py` | Argue opposing positions (Pro/Con) | `google/flan-t5-large` |
| **VerifierAgent** | `task_6_verify_guardrails.py` | NLI factuality + semantic alignment (Cosine) | `facebook/bart-large-mnli` |
| **GuardrailsAgent** | `task_6_verify_guardrails.py` | Redact PII & filter prompt injections | `regex` |
| **MemoryAgent** | `task_7_autotune.py` | Log parameters (α, k, latency, confidence) | `json` |
| **VisualizerAgent** | `task_7_visualize.py` | Plot performance & agent graph | `matplotlib`, `seaborn`, `networkx` |

---

## Tech Stack

| Category | Libraries |
|---|---|
| **PDF Parsing** | `pdfplumber` |
| **Classical NLP** | `spaCy`, `scikit-learn (LDA)` |
| **Embeddings** | `sentence-transformers` |
| **Vector DB** | `Pinecone` |
| **Sparse Retrieval** | `rank-bm25` |
| **Reranking** | `CrossEncoder (sentence-transformers)` |
| **LLM (local)** | `google/flan-t5-large` via HuggingFace |
| **Factuality Check** | `facebook/bart-large-mnli` |
| **Language Detection** | `langdetect` |
| **Text Splitting** | `langchain-text-splitters` |
| **Visualization** | `matplotlib`, `seaborn`, `networkx` |

---

## Project Structure

```
├── src/
│   └── agents/
│       ├── task_1_ingestion.py           # PDF ingestion & chunking
│       ├── task_2_modeling.py            # Preprocessing, LDA, embeddings
│       ├── retriever_agent.py            # Baseline BM25 + Pinecone retriever
│       ├── retriever_experiment_agent.py # Cross-Encoder reranking
│       ├── planner_agent.py              # Query decomposition + language detect
│       ├── summarizer_agent.py           # Citation-aware summarization
│       ├── task_5_debate.py              # Pro/Con debate agents
│       ├── task_6_verify_guardrails.py   # Verifier + guardrails
│       ├── task_7_autotune.py            # Memory + hyperparameter logging
│       └── task_7_visualize.py           # Performance plots
├── data/
│   └── pdfs/                            # Source policy documents (OECD, IMF, UN)
├── results/
│   ├── plots/                           # Generated performance charts
│   ├── classical_output.json            # NLP pipeline output
│   ├── retrieval_ablation.json          # Ablation study results
│   ├── metrics.json                     # Evaluation metrics
│   ├── plan.json                        # Query decomposition plan
│   ├── tuning_log.json                  # Hyperparameter tuning log
│   └── final_policy_brief.txt           # Generated policy brief
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- A [Pinecone](https://www.pinecone.io/) account and API key

### 1. Clone the repository

```bash
git clone https://github.com/Swathi-Giri/Mini-Hackathon-NLP-SAT-.git
cd Mini-Hackathon-NLP-SAT-
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 5. Set up environment variables

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name
```

---

## Running the Pipeline

Run each agent in order:

```bash
# Step 1: Ingest PDFs
python src/agents/task_1_ingestion.py

# Step 2: Preprocess, topic model, and embed
python src/agents/task_2_modeling.py

# Step 3: Run retriever (baseline)
python src/agents/retriever_agent.py

# Step 4: Run retriever experiment (cross-encoder reranking)
python src/agents/retriever_experiment_agent.py

# Step 5: Plan and decompose queries
python src/agents/planner_agent.py

# Step 6: Summarize with citations
python src/agents/summarizer_agent.py

# Step 7: Structured debate
python src/agents/task_5_debate.py

# Step 8: Verify + guardrails
python src/agents/task_6_verify_guardrails.py

# Step 9: Auto-tune & log
python src/agents/task_7_autotune.py

# Step 10: Visualize results
python src/agents/task_7_visualize.py
```

---

## Results & Outputs

After running the pipeline, results are saved to `results/`:

| File | Description |
|---|---|
| `classical_output.json` | Full NLP-processed output for all PDF chunks |
| `retrieval_ablation.json` | BM25 vs Dense vs Hybrid comparison across α values |
| `retrieval_comparison.json` | Side-by-side retrieval method metrics |
| `metrics.json` | Final evaluation scores (latency, confidence, factuality) |
| `plan.json` | Query decomposition plan from PlannerAgent |
| `tuning_log.json` | Hyperparameter sweep log (α, k) |
| `final_policy_brief.txt` | LLM-generated policy brief with citations |
| `plots/factuality_vs_alpha.png` | Factuality score across α values |
| `plots/factuality_vs_k.png` | Factuality score across k values |
| `plots/latency_vs_k_and_alpha.png` | Latency heatmap |
| `plots/rag_performance_vs_alpha.png` | RAG performance comparison |
| `embedding_map.png` | 2D UMAP visualization of document embeddings |
| `retrieval_plot.png` | Retrieval performance chart |

---

## Sample Queries

The pipeline was tested on cross-document policy questions such as:

> *"What are the key differences between OECD and UN recommendations on AI governance?"*

> *"How do IMF fiscal policies align with UN sustainability goals?"*

> *"What role does digital infrastructure play in achieving the SDGs according to the source documents?"*

---

## 👩‍💻 Author

**Swathi Giri** — [GitHub](https://github.com/Swathi-Giri)

*Built for the Mini Hackathon NLP SAT examination.*
