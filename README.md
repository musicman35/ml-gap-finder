# ML Gap Finder

ML Gap Finder is a tool that discovers underexplored combinations of machine learning methods by mining published research. It builds a knowledge graph from arXiv papers, detects gaps where well-studied methods have rarely been combined, retrieves supporting evidence, and generates structured research hypotheses with evaluation plans.

## Architecture

```
Streamlit UI (port 8501)
       |
  FastAPI Backend (port 8000)
       |
  +---------+---------+---------+
  |         |         |         |
Neo4j   PostgreSQL  Qdrant    Redis
(graph)  (metadata)  (vectors) (cache)
```

**Services:**

- **Gap Detector** -- Queries the Neo4j knowledge graph to find method pairs that are individually well-studied but rarely combined for a given task
- **Evidence Retriever** -- Finds papers supporting method-task claims using Neo4j graph traversal and Qdrant semantic search
- **Hypothesis Generator** -- Uses an LLM to produce structured hypotheses with mechanisms, assumptions, and evaluation plans grounded in retrieved evidence
- **Literature Positioner** -- Finds semantically similar papers and traces method lineage to position a proposed approach in existing literature

**LLM Providers:** Switchable between Anthropic Claude and Ollama (local models) via configuration.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.13, Pydantic |
| Frontend | Streamlit |
| Knowledge Graph | Neo4j 5 (with APOC) |
| Metadata Store | PostgreSQL 15 |
| Vector Store | Qdrant |
| Cache | Redis 7 |
| LLM | Anthropic Claude / Ollama |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Package Manager | uv |
| Containerization | Docker Compose |

## Prerequisites

- Python 3.11+ (3.13 recommended)
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose
- An Anthropic API key **or** [Ollama](https://ollama.ai/) running locally

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/musicman35/ml-gap-finder.git
cd ml-gap-finder
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```dotenv
# Choose LLM provider: "anthropic" or "ollama"
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Or use Ollama for local models
# LLM_PROVIDER=ollama
# LLM_OLLAMA_MODEL=gemma3:latest

# Database credentials (defaults match docker-compose.yml)
NEO4J_PASSWORD=neo4j_password
DATABASE_URL=postgresql://mlgap:mlgap_password@localhost:5433/mlgapfinder
```

### 3. Start databases

```bash
docker compose up -d postgres neo4j qdrant redis
```

Wait for all services to become healthy:

```bash
docker compose ps
```

### 4. Run the backend

```bash
uv run uvicorn src.main:app --port 8000 --reload
```

The API auto-initializes PostgreSQL tables on first startup. API docs are available at http://localhost:8000/docs.

### 5. Run the frontend

```bash
uv run streamlit run ui/app.py
```

Open http://localhost:8501 in your browser.

### Full Docker deployment

To run everything in containers (backend + frontend + databases):

```bash
docker compose up -d
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501
- Neo4j Browser: http://localhost:7474

## Data Ingestion

The ingestion pipeline harvests papers and builds the knowledge graph:

1. **arXiv Harvest** -- Fetches ML papers from arXiv (cs.LG, cs.AI, cs.CL, stat.ML)
2. **Semantic Scholar Enrichment** -- Adds citation counts and metadata
3. **Papers With Code Integration** -- Fetches validated method and dataset annotations
4. **Method Extraction** -- Extracts methods from abstracts using LLM or pattern matching
5. **Graph Construction** -- Creates Paper and Method nodes with USES relationships in Neo4j
6. **Embedding Generation** -- Generates vector embeddings for semantic search in Qdrant

### Running ingestion

```bash
# Sample run (100 papers, good for testing)
uv run python scripts/run_ingestion.py --mode=sample --limit=100

# Incremental update (last 7 days)
uv run python scripts/run_ingestion.py --mode=incremental --days=7

# Full ingestion (from 2020 onward)
uv run python scripts/run_ingestion.py --mode=full

# Skip LLM-based extraction (faster, pattern-matching only)
uv run python scripts/run_ingestion.py --mode=sample --no-llm
```

## API Endpoints

### Gap Detection

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/gaps/search` | Search for a specific gap between two methods |
| POST | `/api/v1/gaps/discover` | Auto-discover gaps for a task |
| GET | `/api/v1/gaps/{gap_id}` | Get gap details by ID |

### Evidence Retrieval

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/evidence/search` | Find papers supporting a method-task claim |
| POST | `/api/v1/evidence/validate-citation` | Validate a citation claim |

### Hypothesis Generation

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/hypotheses/generate` | Generate a hypothesis for a gap |
| GET | `/api/v1/hypotheses/{hypothesis_id}` | Get a previously generated hypothesis |
| POST | `/api/v1/hypotheses/{hypothesis_id}/evaluate` | Evaluate a hypothesis (tier 1, 2, or 3) |

### Literature Positioning

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/literature/position` | Position an approach in existing literature |
| POST | `/api/v1/literature/related-work` | Generate a related work outline |

### System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (database connectivity) |
| GET | `/api/v1/status` | API status and configuration |

## Evaluation

Hypotheses are evaluated through a 3-tier system:

- **Tier 1 -- Objective Metrics:** Citation accuracy, gap verification, method extraction F1
- **Tier 2 -- LLM-as-Judge:** Coherence, evidence relevance, and specificity scored 1-5 by the LLM
- **Tier 3 -- Human Calibration:** Compares LLM scores against human annotations using Spearman correlation

```bash
# Run Tier 1 evaluation
uv run python scripts/run_evaluation.py --tier=1

# Run Tier 2 evaluation
uv run python scripts/run_evaluation.py --tier=2

# Run all tiers
uv run python scripts/run_evaluation.py --tier=all --output=report.txt
```

## Project Structure

```
ml-gap-finder/
├── config/
│   └── settings.py              # Pydantic settings (env-based config)
├── scripts/
│   ├── init_db.py               # Database initialization
│   ├── run_ingestion.py         # Data ingestion pipeline
│   └── run_evaluation.py        # Evaluation runner
├── src/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── gaps.py          # Gap detection endpoints
│   │   │   ├── evidence.py      # Evidence retrieval endpoints
│   │   │   ├── hypotheses.py    # Hypothesis generation endpoints
│   │   │   └── literature.py    # Literature positioning endpoints
│   │   └── schemas.py           # Pydantic request/response models
│   ├── db/
│   │   ├── neo4j.py             # Neo4j client
│   │   ├── postgres.py          # PostgreSQL client
│   │   ├── qdrant.py            # Qdrant vector store client
│   │   └── redis.py             # Redis cache client
│   ├── evaluation/
│   │   ├── tier1.py             # Objective metrics evaluator
│   │   ├── tier2.py             # LLM-as-judge evaluator
│   │   └── tier3.py             # Human calibration
│   ├── ingestion/
│   │   ├── arxiv.py             # arXiv paper harvester
│   │   ├── semantic_scholar.py  # Semantic Scholar API client
│   │   ├── papers_with_code.py  # Papers With Code API client
│   │   └── method_extractor.py  # Method extraction from abstracts
│   ├── llm/
│   │   ├── client.py            # LLM client factory (Anthropic/Ollama)
│   │   └── prompts.py           # Prompt templates
│   ├── services/
│   │   ├── gap_detector.py      # Gap detection logic
│   │   ├── evidence_retriever.py # Evidence retrieval logic
│   │   ├── hypothesis_generator.py # Hypothesis generation logic
│   │   └── literature_positioner.py # Literature positioning logic
│   └── main.py                  # FastAPI app entry point
├── tests/
│   ├── conftest.py              # Shared test fixtures
│   ├── test_gap_detector.py     # Gap detector tests
│   ├── test_hypothesis_generator.py # Hypothesis generator tests
│   └── test_llm_client.py       # LLM client tests
├── ui/
│   ├── components/              # Reusable Streamlit components
│   ├── pages/
│   │   ├── 1_Gap_Detection.py
│   │   ├── 2_Evidence_Retrieval.py
│   │   ├── 3_Hypothesis_Generator.py
│   │   └── 4_Literature_Position.py
│   ├── api_client.py            # HTTP client for the backend API
│   └── app.py                   # Streamlit entry point
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── pyproject.toml
```

## Development

### Running tests

```bash
uv run pytest tests/
```

### Linting

```bash
uv run ruff check src/ tests/
```

### Type checking

```bash
uv run mypy src/
```

## License

MIT
