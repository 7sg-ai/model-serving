<!-- HF apps are vLLM-only: set BACKEND_URLS or VLLM_API_URL; no local Transformers. -->
# HuggingFace Streamlit Chat Interface

A chat interface powered by vLLM (OpenAI-compatible) with optional authentication and conversation history.

## Features

- 🤗 Remote vLLM inference
- 🔐 Optional user authentication
- 📝 Conversation history (for logged-in users)
- 🔄 Multiple conversation management
- 👤 Guest mode support
- 🚀 GPU acceleration support

## Prerequisites

- Python 3.8+
- NVIDIA GPU (optional, but recommended for better performance)
- CUDA toolkit (if using GPU)

## Setup

### 1. Install Python Dependencies

```bash
cd hf-streamlit-chat
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your preferred HuggingFace model
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

**Note**: The first run will download the model, which may take some time depending on the model size.

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

- `HF_MODEL_NAME`: HuggingFace model to use (default: meta-llama/Llama-3.1-8B-Instruct)

### Recommended Models

**Small Models (CPU-friendly):**
- `microsoft/DialoGPT-small` - 117M parameters
- `meta-llama/Llama-3.1-8B-Instruct` - 345M parameters (default)
- `facebook/blenderbot-400M-distill` - 400M parameters

**Larger Models (GPU recommended):**
- `microsoft/DialoGPT-large` - 762M parameters
- `meta-llama/Llama-2-7b-chat-hf` - 7B parameters (requires HF token)
- `mistralai/Mistral-7B-Instruct-v0.1` - 7B parameters

**Note**: Some models require accepting terms on HuggingFace and using an access token.

## Using Gated Models

For gated models (like Llama 2), you need to:

1. Accept the model terms on HuggingFace
2. Create an access token at https://huggingface.co/settings/tokens
3. Login via CLI:
```bash
huggingface-cli login
```

## Performance Tips

### GPU Acceleration

The app automatically uses GPU if available. Check the sidebar to see which device is being used.

### Memory Management

For large models, you may need to:
- Use a smaller model
- Reduce `max_length` in the generation parameters
- Use model quantization (add to requirements: `bitsandbytes`)

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
1. Use a smaller model
2. Restart the application
3. Clear the Streamlit cache: `streamlit cache clear`

### Slow Response Times

- Ensure you're using GPU if available
- Try a smaller model
- Reduce the conversation history length

### Model Download Issues

If model download fails:
```bash
# Pre-download the model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')"
```

## Model Storage

Downloaded models are cached in:
- Linux/Mac: `~/.cache/huggingface/`
- Windows: `C:\Users\<username>\.cache\huggingface\`




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
NODE_ID=hf-chat-1
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
