# HuggingFace IDE Coding Assistant

An OpenAI-compatible API server that fronts **vLLM** (HF model weights) for IDEs like Cline and Cursor.
This process does **not** load Transformers weights; configure `BACKEND_URLS` / `VLLM_API_URL`.

## Features

- 🔌 OpenAI-compatible API endpoints
- 🚀 Powered by **vLLM** (OpenAI-compatible HTTP)
- 💻 Works with Cline, Cursor, and other OpenAI-compatible tools
- 🔑 API key authentication
- 📝 Optimized for complicated multi-file coding (context trim + cache)
- 🚀 GPU acceleration on the vLLM servers (TP/replica/hybrid)
- 🧠 1M-token context budget with automatic trim to model capacity
- ⚡ In-memory response cache + conversation memory (KV cache on generate)
- 🥮 **Mooncake** L2 cache auto-enabled for Moonshot / Kimi models


## Prerequisites

- **vLLM** OpenAI server(s) reachable via `BACKEND_URLS` or `VLLM_API_URL`
  (see `deploy/docker-compose.vllm.yml` / `deploy/scripts/generate_vllm_compose.py`)

- Python 3.8+
- NVIDIA GPU (optional, but recommended for better performance)
- CUDA toolkit (if using GPU)
- Cline or Cursor IDE extension

## Setup

### 1. Start vLLM

```bash
# single replica
docker compose -f deploy/docker-compose.vllm.yml up -d vllm-0
# or multi replica/hybrid
# python deploy/scripts/generate_vllm_compose.py --include-apps && docker compose -f deploy/docker-compose.vllm.generated.yml up -d
```

### 2. Install Python Dependencies

```bash
cd hf-ide-assistant
pip install -r requirements.txt
```

**Note**: The first run will download the model, which may take some time.

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your preferred model
```

### 4. Run the Server

```bash
python server.py
```

The server will start on port 8081 and display configuration details.

## IDE Configuration

### For Cline (VS Code Extension)

> **Streaming:** Cline sends `stream: true` by default. This server proxies OpenAI SSE
> (`text/event-stream`) from the upstream backend. Non-stream JSON still works.

1. Open VS Code Settings
2. Search for "Cline"
3. Configure:
   - **API Provider**: OpenAI Compatible
   - **Base URL**: `http://localhost:8081/v1`
   - **API Key**: `hf-coding-assistant-key` (or your custom key from .env)
   - **Model**: `moonshotai/Kimi-K3` (or your configured model)


### For Cursor

1. Open Cursor Settings (Cmd/Ctrl + ,)
2. Go to "Models" section
3. Add Custom Model:
   - **Provider**: OpenAI Compatible
   - **Base URL**: `http://localhost:8081/v1`
   - **API Key**: `hf-coding-assistant-key`
   - **Model ID**: `moonshotai/Kimi-K3`


### For Continue (VS Code Extension)

Edit `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "HuggingFace Kimi-K3",
      "provider": "openai",
      "model": "moonshotai/Kimi-K3",
      "apiBase": "http://localhost:8081/v1",
      "apiKey": "hf-coding-assistant-key"

    }
  ]
}
```

## API Endpoints

### List Models
```bash
GET /v1/models
Authorization: Bearer hf-coding-assistant-key
```

### Chat Completions
```bash
POST /v1/chat/completions
Authorization: Bearer hf-coding-assistant-key
Content-Type: application/json

{
  "model": "moonshotai/Kimi-K3",
  "messages": [
    {"role": "user", "content": "Write a Python function to sort a list"}
  ]

}
```

### Text Completions
```bash
POST /v1/completions
Authorization: Bearer hf-coding-assistant-key
Content-Type: application/json

{
  "model": "moonshotai/Kimi-K3",
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

- `HF_MODEL_NAME` / `VLLM_MODEL`: model id **must match** vLLM `--model` (default: meta-llama/Llama-3.1-8B-Instruct)
- `BACKEND_URLS` or `VLLM_API_URL`: required vLLM chat completions endpoint(s)
- `MODEL_DEPLOY_MODE`: `hybrid` | `replica` | `sharded` (placement metadata; GPUs/TP set on vLLM)

- `API_KEY`: API key for authentication (change in production!)
- `MAX_TOKENS`: Maximum tokens per response (default: **32768**)
- `TEMPERATURE`: Response randomness 0.0-1.0 (default: **0.3** — precise coding)
- `CONTEXT_WINDOW`: Context budget (default: **1048576** / 1M tokens; trimmed to model max)
- `CACHE_ENABLED`: Enable response caching (default: `true`)
- `CACHE_MAX_SIZE`: LRU cache entries (default: `256`)
- `CACHE_TTL_SECONDS`: Cache entry lifetime (default: `3600`)
- `MEMORY_MAX_CONVERSATIONS`: In-memory conversation sessions (default: `128`)

### Understanding Context Window vs Max Tokens

**CONTEXT_WINDOW**: The maximum total size (in tokens) of the conversation history budget. This includes:
- System messages
- Previous user messages
- Previous assistant responses
- Current user message

The server also reports an **effective** context window (`effective_context_window` on `/health` and `/v1/models`) which is `min(CONTEXT_WINDOW, model_max_length)`.

**MAX_TOKENS**: The maximum number of tokens the model can generate in a single response. This is per-request and controls the length of each individual answer. Actual generation is also capped by the model's native context.

**Defaults (complicated coding)**:
- `CONTEXT_WINDOW=1048576` — **1M-token** budget for large codebases
- `MAX_TOKENS=32768` — long multi-file generations when the model supports it
- `TEMPERATURE=0.3` — more deterministic code output

### Caching & Conversation Memory

The server speeds up repeated and follow-up IDE requests with several layers:

1. **Response cache (L1)** — identical `(model, messages, temperature, max_tokens)` requests return the cached completion instantly (`cached: true` in the JSON body). LRU + TTL eviction.
2. **Mooncake response L2 (optional)** — when the model id looks like Moonshot/Kimi, full **completion JSON** may also be stored in Mooncake as an app-level L2. This is **not** vLLM GPU KV. Engine-side hierarchical KV (GPU working set + Mooncake store on overflow/share) is configured on the **vLLM** deployment.
3. **Conversation memory** — rolling per-session history (via `X-Conversation-ID` header, or `user` / `conversation_id` body fields). History is trimmed to the effective context while reserving room for `MAX_TOKENS`.
4. **HF KV cache** — `model.generate(..., use_cache=True)` for faster token-by-token decoding.

#### Enabling Mooncake (Moonshot / Kimi)

```bash
# Install (Linux + CUDA; see Mooncake docs for non-CUDA / CUDA 13 wheels)
pip install mooncake-transfer-engine

# Start mooncake_master (see https://kvcache-ai.github.io/Mooncake/)
# then configure .env:
MOONCAKE_CACHE=auto          # auto | true | false
MOONCAKE_MASTER=127.0.0.1:50051
MOONCAKE_PROTOCOL=tcp
```

`/health` reports `cache.backend` as `mooncake+l1` when connected, or `l1` with `mooncake_error` when falling back.


```bash
# Health includes cache/memory stats and effective context
curl http://localhost:8081/health

# Clear response cache
curl -X DELETE -H "Authorization: Bearer hf-coding-assistant-key" \
  http://localhost:8081/v1/cache

# Clear cache + conversation memory
curl -X DELETE -H "Authorization: Bearer hf-coding-assistant-key" \
  "http://localhost:8081/v1/cache?memory=true"
```

### Adjusting Context and Token Limits

Configure these in the `.env` file:

```bash
# Complicated coding (default) — large repos, multi-file edits
MAX_TOKENS=32768
CONTEXT_WINDOW=1048576
TEMPERATURE=0.3

# Balanced (good for 7B instruct models on GPU)
MAX_TOKENS=4096
CONTEXT_WINDOW=8192

# Faster / lighter (small local models only)
MAX_TOKENS=1024
CONTEXT_WINDOW=2048
```

**Important Notes**:
- MAX_TOKENS should always be less than CONTEXT_WINDOW
- These are server-side defaults; IDEs can override them per-request
- Larger values require more memory; adjust based on your hardware
- **moonshotai/Kimi-K3** natively supports a **1M** context window, matching the default `CONTEXT_WINDOW`
- Smaller HF models have lower native limits — the server reports `effective_context_window` on `/health`

## Recommended Models for Coding

**Default:**
- `moonshotai/Kimi-K3` - Moonshot Kimi K3 (2.8T MoE / ~104B active, **1M context**) — best for complicated multi-file coding

**Lighter alternatives:**
- `mistralai/Mistral-7B-Instruct-v0.1` - 7B parameters
- `meta-llama/Llama-2-7b-chat-hf` - 7B parameters (requires HF token)
- `Salesforce/codegen-2B-mono` - Code generation focused
- `microsoft/DialoGPT-medium` - Small conversational baseline (not ideal for coding)


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
2. Check the port is not in use: `lsof -i :8081`

### Authentication Errors

1. Check API key matches in `.env` and IDE configuration
2. Ensure Authorization header is being sent

### Model Loading Issues

If the model fails to load:
```bash
# Pre-download the model (Kimi-K3 is very large — ensure disk + GPU capacity)
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('moonshotai/Kimi-K3', trust_remote_code=True, torch_dtype='auto', device_map='auto')"
```

### Out of Memory Errors

1. Ensure multi-GPU / enough VRAM for Kimi-K3, or set `HF_MODEL_NAME` to a smaller model
2. Use `HF_TORCH_DTYPE=bfloat16` (default on CUDA) or `float16`
3. Close other applications


### Slow Responses

- Use GPU if available
- Try a smaller model
- Reduce conversation history length
- Ensure `CACHE_ENABLED=true` so repeated prompts hit the cache
- Reuse the same `X-Conversation-ID` for multi-turn sessions

## Performance Tips

- **GPU Acceleration**: The app automatically uses GPU if available
- **Model Size**: Start with smaller models and scale up based on your hardware
- **Memory Management**: Monitor memory usage, especially with larger models
- **Conversation History**: Clear conversation history periodically for better performance (`DELETE /v1/cache?memory=true`)
- **Response Cache**: Keep caching enabled for iterative IDE workflows
- **Temperature**: Use 0.1–0.3 for more stable code edits


## Security Notes

- **Change the API_KEY** in production
- Use HTTPS in production environments
- Consider implementing rate limiting
- Restrict access to localhost or trusted networks

## Port Configuration

This application runs on port **8081** by default (different from NIM version on 8080). You can change this in `server.py` if needed.

## Comparison with NIM Version

**HuggingFace Version:**
- ✅ Fully local, no external dependencies
- ✅ Free and open source
- ✅ Wide model selection
- ⚠️ Requires more local resources
- ⚠️ May be slower without GPU

**NIM Version:**
- ✅ Optimized inference performance
- ✅ Better GPU utilization
- ✅ Enterprise-grade models
- ⚠️ Requires NVIDIA GPU
- ⚠️ Requires NGC account


## NeMo Switchyard (multi-model + escalation)

Route IDE traffic across several HF models (local weights and/or remote OpenAI-compatible backends), each with independent deployment parameters (GPUs, `device_map`, dtype, TP, …).

```bash
# .env
SWITCHYARD_ENABLED=true
SWITCHYARD_STRATEGY=escalation
SWITCHYARD_ROUTE_ID=switchyard/agent
SWITCHYARD_CONFIG=./switchyard-models.example.json
```

| IDE model field | Behavior |
|-----------------|----------|
| `switchyard/agent` | Escalation route (weak first, latch to strong) |
| Model id from config | Direct generation on that model |

```bash
curl -H "Authorization: Bearer $API_KEY" http://localhost:8081/v1/models
curl -X DELETE -H "Authorization: Bearer $API_KEY" \
  -H "X-Conversation-ID: my-session" \
  http://localhost:8081/v1/switchyard/session
```

Full reference: [`shared/switchyard/README.md`](../shared/switchyard/README.md).

## Multi-Node Deployment

This app supports horizontal multi-node deployment via `shared.multinode`.


### Quick enable

```bash
# In .env (see also deploy/env.multinode.example)
MULTI_NODE=true
CLUSTER_NAME=model-serving
NODE_ID=hf-ide-1
HOST=0.0.0.0
PORT=8081
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
