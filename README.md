# AI IDE Backend

FastAPI backend with LangChain agents, Ollama LLM, and Supabase vector store for RAG.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with `llama3.1:8b` model

## Setup

```bash
# Create and activate virtual environment
python -m venv ~/.venvs/ai-ide-backend
source ~/.venvs/ai-ide-backend/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file with:

```
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
OLLAMA_BASE_URL=http://localhost:11434
```

Pull the Ollama model:

```bash
ollama pull llama3.1:8b
```

## Run

```bash
source ~/.venvs/ai-ide-backend/bin/activate
uvicorn main:app --reload --port 8000
```

The API will be available at [http://localhost:8000](http://localhost:8000).

Health check: [http://localhost:8000/health](http://localhost:8000/health)

## API Routes

- `/chat` — AI chat with agent routing
- `/embed` — Embed documents into vector store
- `/embeddings` — Query embeddings
- `/terminal` — Terminal command execution
