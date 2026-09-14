# NIM IDE Coding Assistant

An OpenAI-compatible API server that connects NIM models to IDEs like Cline and Cursor for coding assistance.

## Features

- 🔌 OpenAI-compatible API endpoints
- 🤖 Powered by NVIDIA NIM models
- 💻 Works with Cline, Cursor, and other OpenAI-compatible tools
- 🔑 API key authentication
- 📝 Optimized for complicated multi-file coding tasks
- 🧠 1M-token context window with automatic history trimming
- ⚡ In-memory response cache + conversation memory for faster follow-ups

## Prerequisites

- Python 3.8+
- Docker (for running NIM container)
- NVIDIA GPU with Docker support (recommended)
- Cline or Cursor IDE extension

## Setup

### 1. Start NIM Container

```bash
docker run -d \
  --gpus all \
  --name nim-llama \
  -p 8000:8000 \
  -e NGC_API_KEY=your_ngc_api_key \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

### 2. Install Python Dependencies

```bash
cd nim-ide-assistant
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Run the Server

```bash
python server.py
```

The server will start on port 8080 and display configuration details.

## IDE Configuration

### For Cline (VS Code Extension)

> **Streaming:** Cline sends `stream: true` by default. This server proxies OpenAI SSE
> (`text/event-stream`) from the upstream backend. Non-stream JSON still works.

1. Open VS Code Settings
2. Search for "Cline"
3. Configure:
   - **API Provider**: OpenAI Compatible
   - **Base URL**: `http://localhost:8080/v1`
   - **API Key**: `nim-coding-assistant-key` (or your custom key from .env)
   - **Model**: `meta/llama-3.1-8b-instruct` (or your configured model)

### For Cursor

1. Open Cursor Settings (Cmd/Ctrl + ,)
2. Go to "Models" section
3. Add Custom Model:
   - **Provider**: OpenAI Compatible
   - **Base URL**: `http://localhost:8080/v1`
   - **API Key**: `nim-coding-assistant-key`
   - **Model ID**: `meta/llama-3.1-8b-instruct`

### For Continue (VS Code Extension)

Edit `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "NIM Llama",
      "provider": "openai",
      "model": "meta/llama-3.1-8b-instruct",
      "apiBase": "http://localhost:8080/v1",
      "apiKey": "nim-coding-assistant-key"
    }
  ]
}
```

## API Endpoints

### List Models
```bash
GET /v1/models
Authorization: Bearer nim-coding-assistant-key
```

### Chat Completions
```bash
POST /v1/chat/completions
Authorization: Bearer nim-coding-assistant-key
Content-Type: application/json

{
  "model": "meta/llama-3.1-8b-instruct",
  "messages": [
    {"role": "user", "content": "Write a Python function to sort a list"}
  ]
}
```

### Text Completions
```bash
POST /v1/completions
Authorization: Bearer nim-coding-assistant-key
Content-Type: application/json

{
  "model": "meta/llama-3.1-8b-instruct",
  "prompt": "def fibonacci(n):",
  "max_tokens": 100
}
```

### Health Check
```bash
GET /health
```

## Configuration

Edit `.env` file:

- `NIM_API_URL`: URL of your NIM API endpoint
- `NIM_MODEL_NAME`: Model name to use
- `API_KEY`: API key for authentication (change in production!)
- `MAX_TOKENS`: Maximum tokens per response (default: **32768**)
- `TEMPERATURE`: Response randomness 0.0-1.0 (default: **0.3** — precise coding)
- `CONTEXT_WINDOW`: Maximum context window size (default: **1048576** / 1M tokens)
- `CACHE_ENABLED`: Enable response caching (default: `true`)
- `CACHE_MAX_SIZE`: LRU cache entries (default: `256`)
- `CACHE_TTL_SECONDS`: Cache entry lifetime (default: `3600`)
- `MEMORY_MAX_CONVERSATIONS`: In-memory conversation sessions (default: `128`)
- `REQUEST_TIMEOUT`: Upstream NIM timeout seconds (default: `300`)

### Understanding Context Window vs Max Tokens

**CONTEXT_WINDOW**: The maximum total size (in tokens) of the conversation history that the model can "see" at once. This includes:
- System messages
- Previous user messages
- Previous assistant responses
- Current user message

**MAX_TOKENS**: The maximum number of tokens the model can generate in a single response. This is per-request and controls the length of each individual answer.

**Defaults (complicated coding)**:
- `CONTEXT_WINDOW=1048576` — up to **1M tokens** of codebase / conversation context
- `MAX_TOKENS=32768` — long multi-file generations and refactors per reply
- `TEMPERATURE=0.3` — more deterministic code output

### Caching & Conversation Memory

The server speeds up repeated and follow-up IDE requests with two layers:

1. **Response cache** — identical `(model, messages, temperature, max_tokens)` requests return the cached completion instantly (`cached: true` in the JSON body). LRU + TTL eviction.
2. **Conversation memory** — rolling per-session history (via `X-Conversation-ID` header, or `user` / `conversation_id` body fields). History is trimmed to fit `CONTEXT_WINDOW` while reserving room for `MAX_TOKENS`.

```bash
# Health includes cache/memory stats
curl http://localhost:8080/health

# Clear response cache
curl -X DELETE -H "Authorization: Bearer nim-coding-assistant-key" \
  http://localhost:8080/v1/cache

# Clear cache + conversation memory
curl -X DELETE -H "Authorization: Bearer nim-coding-assistant-key" \
  "http://localhost:8080/v1/cache?memory=true"
```

### Adjusting Context and Token Limits

Configure these in the `.env` file:

```bash
# Complicated coding (default) — large repos, multi-file edits
MAX_TOKENS=32768
CONTEXT_WINDOW=1048576
TEMPERATURE=0.3

# Balanced
MAX_TOKENS=8192
CONTEXT_WINDOW=131072

# Faster / lighter (less GPU memory)
MAX_TOKENS=2048
CONTEXT_WINDOW=8192
```

**Important Notes**:
- MAX_TOKENS should always be less than CONTEXT_WINDOW
- These are server-side defaults; IDEs can override them per-request
- Larger values require more GPU memory
- The model's native context limit may be lower than your setting — the server still trims prompts to `CONTEXT_WINDOW`

## Recommended NIM Models for Coding

- **meta/llama-3.1-8b-instruct** - Good balance of speed and quality
- **meta/llama-3.1-70b-instruct** - Higher quality, requires more resources
- **codellama/CodeLlama-34b-Instruct-hf** - Specialized for coding tasks

## Usage Examples

### Code Generation
Ask Cline/Cursor: "Create a REST API with Flask that has CRUD endpoints for a todo list"

### Code Explanation
Select code and ask: "Explain what this function does"

### Debugging
Paste error and ask: "Help me fix this error"

### Refactoring
Select code and ask: "Refactor this to be more efficient"

## Troubleshooting

### Connection Refused

1. Ensure the server is running: `python server.py`
2. Check the port is not in use: `lsof -i :8080`
3. Verify NIM container is running: `docker ps | grep nim`

### Authentication Errors

1. Check API key matches in `.env` and IDE configuration
2. Ensure Authorization header is being sent

### Slow Responses

1. Use a GPU for better performance
2. Try a smaller model
3. Reduce `max_tokens` in requests
4. Ensure `CACHE_ENABLED=true` so repeated prompts hit the cache
5. Reuse the same `X-Conversation-ID` so memory can avoid re-sending full history from the client

### Model Not Found

1. Verify NIM container is running with the correct model
2. Check `NIM_MODEL_NAME` in `.env` matches the container model

## Security Notes

- **Change the API_KEY** in production
- Use HTTPS in production environments
- Consider implementing rate limiting
- Restrict access to localhost or trusted networks

## Performance Tips

- Use GPU for significantly faster inference
- Adjust `max_tokens` based on your needs
- Consider using a smaller model for faster responses
- Keep caching enabled for iterative IDE workflows (retry / re-ask same prompt)
- Lower `TEMPERATURE` (0.1–0.3) for more stable code edits


## Port Configuration

This application runs on port **8080** by default. You can change this in `server.py` if needed.


## NeMo Switchyard (multi-model + escalation)

Route IDE traffic across several NIM backends with independent deployment parameters.

```bash
# .env
SWITCHYARD_ENABLED=true
SWITCHYARD_STRATEGY=escalation
SWITCHYARD_ROUTE_ID=switchyard/agent
SWITCHYARD_CONFIG=./switchyard-models.example.json
```

Or set tiers via env (`SWITCHYARD_WEAK_*`, `SWITCHYARD_STRONG_*`, `SWITCHYARD_JUDGE_*`) — each can have its own URL, GPUs, TP size, max_tokens, etc.

| IDE model field | Behavior |
|-----------------|----------|
| `switchyard/agent` | Escalation route (weak first, latch to strong) |
| `meta/llama-3.1-8b-instruct` (etc.) | Direct to that model’s backend |

```bash
curl -H "Authorization: Bearer $API_KEY" http://localhost:8080/v1/models
# Reset escalation latch for a session
curl -X DELETE -H "Authorization: Bearer $API_KEY" \
  -H "X-Conversation-ID: my-session" \
  http://localhost:8080/v1/switchyard/session
```

Full reference: [`shared/switchyard/README.md`](../shared/switchyard/README.md).

## Multi-Node Deployment

This app supports horizontal multi-node deployment via `shared.multinode`.


### Quick enable

```bash
# In .env (see also deploy/env.multinode.example)
MULTI_NODE=true
CLUSTER_NAME=model-serving
NODE_ID=nim-ide-1
HOST=0.0.0.0
PORT=8080
PUBLIC_URL=https://models.example.com
SHARED_DATA_DIR=/data/model-serving
REDIS_URL=redis://redis:6379/0
USE_REDIS=true
SECRET_KEY=replace-with-long-random-string
# Optional: multiple model backends
BACKEND_URLS=http://nim-a:8000/v1/chat/completions,http://nim-b:8000/v1/chat/completions
BACKEND_STRATEGY=round_robin
```

### Architecture notes

| Concern | Approach |
|--------|----------|
| Shared auth / chat history | Mount the same `SHARED_DATA_DIR` on every node (NFS/EFS) or set `AUTH_DB_PATH` / `CHAT_DB_PATH` |
| Load balancing | Put nginx/ALB/ingress in front; enable **sticky sessions** for Streamlit & WebRTC |
| Socket.IO fan-out | Set `REDIS_URL` + `USE_REDIS=true` so all voice nodes share a message queue |
| Model backends | `BACKEND_URLS` round-robins NIM / OpenAI-compatible upstreams |
| Health | `GET /health` (APIs) or sidebar "Cluster / node" (Streamlit) reports `node_id` and peers |

### Docker Compose reference

```bash
# From repo root — example scales NIM IDE behind nginx + Redis
docker compose -f deploy/docker-compose.multinode.yml up -d
```

Full variable reference: [`shared/multinode/README.md`](../shared/multinode/README.md) and [`deploy/env.multinode.example`](../deploy/env.multinode.example).
