# NIM Streamlit Chat Interface

A chat interface powered by NVIDIA NIM (NVIDIA Inference Microservices) with optional authentication and conversation history.

## Features

- 💬 Interactive chat interface using Streamlit
- 🔐 Optional user authentication
- 📝 Conversation history (for logged-in users)
- 🔄 Multiple conversation management
- 👤 Guest mode support

## Prerequisites

- Python 3.8+
- Docker (for running NIM container)
- NVIDIA GPU with Docker support (recommended)

## Setup

### 1. Start NIM Container

First, you need to run a NIM container. Example using Llama 3.1:

```bash
# Pull and run NIM container
docker run -d \
  --gpus all \
  --name nim-llama \
  -p 8000:8000 \
  -e NGC_API_KEY=your_ngc_api_key \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

Get your NGC API key from: https://catalog.ngc.nvidia.com/

### 2. Install Python Dependencies

```bash
cd nim-streamlit-chat
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your NIM API URL and model name
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

### Login Options

1. **Login**: Use existing credentials
2. **Register**: Create a new account
3. **Guest Mode**: Use without authentication (no history saved)

### Chat Features

- Type your message in the chat input
- View conversation history in the sidebar (logged-in users)
- Start new conversations with the "New Conversation" button
- Load previous conversations from the sidebar

## Configuration

Edit `.env` file:

- `NIM_API_URL`: URL of your NIM API endpoint (default: http://localhost:8000/v1/chat/completions)
- `NIM_MODEL_NAME`: Model name to use (default: meta/llama-3.1-8b-instruct)

## Troubleshooting

### NIM Container Not Running

Check if the container is running:
```bash
docker ps | grep nim
```

View container logs:
```bash
docker logs nim-llama
```

### Connection Errors

Ensure the NIM API URL in `.env` matches your container's exposed port.

## Available NIM Models

Visit [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/) for available NIM models:
- Llama 3.1 (8B, 70B, 405B)
- Mistral
- Mixtral
- And more...




## Internet search + URL fetch

When a user message **asks to search** (e.g. "search the web for …", "look up …", or `/search …`),
the app queries **DuckDuckGo HTML** (no API key), optionally fetches the top result pages, and
injects titles/URLs/snippets into the model context.

When a message contains explicit `http://` or `https://` URLs, those pages are still fetched and logged
(see original Internet access tool behavior below).

```bash
WEB_SEARCH_ENABLED=true
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_FETCH_TOP=2
WEB_SEARCH_FETCH_PAGES=true
```

Requires outbound HTTPS egress from the deployment. Disable with `WEB_SEARCH_ENABLED=false`.

## Internet access tool

When a user message contains `http://` or `https://` URLs, the app:

1. Fetches each page
2. Converts HTML to readable text for the model (may truncate for the prompt)
3. **Always logs the URL and the full fetched body** under `WEB_ACCESS_LOG_DIR` (default `shared/database/web_access_logs/`)

### Logs (full content, not just the model snippet)

| Artifact | Location |
|----------|----------|
| Fixed index + full body stream (no rotation) | `WEB_ACCESS_LOG_DIR/<app>-web-access.log` |
| Per-URL full content files | `WEB_ACCESS_LOG_DIR/pages/<app>/*.txt` |

Each per-URL file includes the URL, status, title, SHA-256, and the **complete** extracted text.

**Log full ⇒ web access terminates** (no rotating backups). Clear logs or raise `WEB_ACCESS_LOG_MAX_SIZE` (e.g. `2TB`) to resume after restart.

### Config

```bash
WEB_ACCESS_ENABLED=true
WEB_ACCESS_LOG_DIR=shared/database/web_access_logs
WEB_ACCESS_LOG_MAX_SIZE=1TB              # e.g. 50GB, 1TB, 2GiB; access stops when full
WEB_ACCESS_TIMEOUT=20
WEB_ACCESS_MAX_BYTES=2000000
WEB_ACCESS_MAX_PROMPT_CHARS=12000
WEB_ACCESS_MAX_URLS=5
```

Disable with `WEB_ACCESS_ENABLED=false`.

## Multi-Node Deployment

This app supports horizontal multi-node deployment via `shared.multinode`.

### Quick enable

```bash
# In .env (see also deploy/env.multinode.example)
MULTI_NODE=true
CLUSTER_NAME=model-serving
NODE_ID=nim-chat-1
HOST=0.0.0.0
PORT=8501
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
