# Personal Database

A Retrieval-Augmented Generation (RAG) application for personal knowledge management. Scrape, process, chunk, embed, and query web content through a chat interface powered by LLMs.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Runtime Management** | [mise](https://mise.jdx.dev/) — Python 3.11, Node.js 20 |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/), Uvicorn, Pydantic |
| **Vector Database** | [Weaviate](https://weaviate.io/) (via Docker Compose) |
| **Embeddings** | [sentence-transformers](https://www.sbert.net/) (all-MiniLM-L6-v2) |
| **LLM** | [OpenRouter](https://openrouter.ai/) (Anthropic Claude, fallback extractive answers) |
| **Frontend** | React 18, Material-UI, react-markdown |
| **Scraping** | requests, BeautifulSoup 4, lxml |
| **Text Processing** | NLTK, spaCy |
| **Testing** | pytest, pytest-asyncio, Playwright |

### Prerequisites

Install [mise](https://mise.jdx.dev/getting-started.html) for runtime version management:

```bash
# macOS/Linux
curl https://mise.run | sh

# Then activate (add to your shell profile)
echo 'eval "$(mise activate bash)"' >> ~/.bashrc  # or ~/.zshrc
```

## Setup

```bash
# Install runtimes via mise
mise install

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (frontend)
cd src/web && npm install && cd ../..

# Start Weaviate
docker compose up -d
```

## Running

### Backend (FastAPI API)

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is available at `http://localhost:8000`. Interactive docs at `/docs`.

### Frontend (React)

```bash
cd src/web && npm start
```

The frontend is available at `http://localhost:3000`.

### Processing Scripts (CLI)

Run scripts directly from the `scripts/` directory:

```bash
# Scrape URLs
python scripts/scraper.py https://example.com -o output/

# Parse scraped content
python scripts/parser.py output/ -o parsed/

# Chunk documents
python scripts/chunker.py parsed/ --strategy semantic -o chunks/

# Generate embeddings and store in Weaviate
python scripts/embedder.py chunks/
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest/url` | Ingest a single URL (scrape → parse → chunk → embed) |
| `POST` | `/ingest/urls` | Ingest multiple URLs in batch |
| `POST` | `/ingest/upload` | Upload and process files (.json, .txt) |
| `POST` | `/query/` | Query the knowledge base with RAG |
| `GET` | `/health/` | Service health status |
| `GET` | `/health/ping` | Basic connectivity check |

## Testing

```bash
# Run all tests
make test

# Backend tests only
make test-backend

# Backend unit tests
make test-backend-unit

# Frontend unit tests
make test-frontend-unit

# Frontend E2E tests
make test-frontend-e2e

# With coverage
make test-coverage
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                       │
│              (Material-UI Chat Interface)               │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
                       ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │  /ingest/*   │  │   /query/*   │  │  /health/*    │ │
│  │  (Scrape)    │  │   (RAG)      │  │  (Status)     │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────────┘ │
│         │                 │                            │
│         ▼                 ▼                            │
│  ┌─────────────────────────────────┐                   │
│  │        RAG Service              │                   │
│  │  ┌────────────┐ ┌─────────────┐ │                   │
│  │  │ Vector DB  │ │ Embedding   │ │                   │
│  │  │ Service    │ │ Service     │ │                   │
│  │  └─────┬──────┘ └──────┬──────┘ │                   │
│  └────────┼───────────────┼────────┘                   │
└───────────┼───────────────┼────────────────────────────┘
            │               │
            ▼               ▼
┌───────────────────┐  ┌──────────────────────────────────┐
│    Weaviate       │  │  sentence-transformers           │
│  (Vector Store)   │  │  (Embedding Model)               │
└───────────────────┘  └──────────────────────────────────┘
```

### Ingestion Pipeline

1. **Scrape** (`scripts/scraper.py`) — Fetches web content via requests + BeautifulSoup
2. **Parse** (`scripts/parser.py`) — Cleans HTML, extracts text and metadata
3. **Chunk** (`scripts/chunker.py`) — Splits documents using character, token, semantic, or pattern-based strategies
4. **Embed** (`scripts/embedder.py`) — Generates embeddings with sentence-transformers and stores in Weaviate

### Query Pipeline

1. User submits a question via the chat interface
2. The question is embedded using the same model
3. Weaviate retrieves the most similar document chunks
4. Retrieved context + question are sent to the LLM (via OpenRouter)
5. The generated answer is returned with source citations

## Configuration

Copy `configs/default.yaml` and adjust settings as needed. Set environment variables for API keys:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

