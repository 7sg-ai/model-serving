# NeMo Switchyard Integration

Multi-model routing for **nim-ide-assistant** and **hf-ide-assistant**, aligned with
[NVIDIA NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard).

## Features

| Feature | Description |
|--------|-------------|
| **Multi-model selection** | Declare several models; clients pick a model id or the virtual route id |
| **Escalation router** | Weak → judge → strong latch (Switchyard `llm_classifier` + `mode = "escalation"`) |
| **Per-model deployment** | Independent backend URLs, GPUs, TP/PP, dtype, timeouts, max_tokens, etc. |
| **External proxy** | Optional forward to official `switchyard-server` |
| **routes.toml export** | Generate config for the Rust Switchyard binary |

## Quick enable (env)

```bash
SWITCHYARD_ENABLED=true
SWITCHYARD_STRATEGY=escalation   # escalation | capability | random | passthrough | external
SWITCHYARD_ROUTE_ID=switchyard/agent

# Weak / strong / judge with independent backends + deploy params
SWITCHYARD_WEAK_MODEL=meta/llama-3.1-8b-instruct
SWITCHYARD_WEAK_URL=http://localhost:8000/v1/chat/completions
SWITCHYARD_WEAK_CUDA_VISIBLE_DEVICES=0
SWITCHYARD_WEAK_MAX_TOKENS=8192

SWITCHYARD_STRONG_MODEL=meta/llama-3.1-70b-instruct
SWITCHYARD_STRONG_URL=http://localhost:8001/v1/chat/completions
SWITCHYARD_STRONG_CUDA_VISIBLE_DEVICES=1,2,3,4
SWITCHYARD_STRONG_TENSOR_PARALLEL_SIZE=4
SWITCHYARD_STRONG_MAX_TOKENS=32768

SWITCHYARD_JUDGE_MODEL=meta/llama-3.1-8b-instruct
SWITCHYARD_JUDGE_URL=http://localhost:8000/v1/chat/completions

# Escalation tuning (Switchyard defaults)
SWITCHYARD_ESCALATION_CONFIRMATIONS=2
SWITCHYARD_ESCALATION_TURN_WINDOW=28
SWITCHYARD_ESCALATION_MSG_CHARS=500
```

## JSON config file

```bash
SWITCHYARD_CONFIG=./switchyard-models.json
```

```json
{
  "enabled": true,
  "strategy": "escalation",
  "route_id": "switchyard/agent",
  "expose_models": true,
  "allow_model_select": true,
  "escalation": {
    "confirmations": 2,
    "recent_turn_window": 28,
    "window_message_chars": 500
  },
  "models": [
    {
      "name": "weak",
      "id": "meta/llama-3.1-8b-instruct",
      "role": "weak",
      "max_tokens": 8192,
      "temperature": 0.3,
      "context_window": 131072,
      "deployment": {
        "backend_url": "http://nim-8b:8000/v1/chat/completions",
        "cuda_visible_devices": "0",
        "gpu_count": 1,
        "nim_image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:latest"
      }
    },
    {
      "name": "strong",
      "id": "meta/llama-3.1-70b-instruct",
      "role": "strong",
      "max_tokens": 32768,
      "context_window": 131072,
      "deployment": {
        "backend_urls": [
          "http://nim-70b-a:8000/v1/chat/completions",
          "http://nim-70b-b:8000/v1/chat/completions"
        ],
        "backend_strategy": "round_robin",
        "tensor_parallel_size": 4,
        "cuda_visible_devices": "0,1,2,3",
        "gpu_count": 4
      }
    },
    {
      "name": "judge",
      "id": "meta/llama-3.1-8b-instruct",
      "role": "judge",
      "max_tokens": 1024,
      "deployment": {
        "backend_url": "http://nim-8b:8000/v1/chat/completions"
      }
    }
  ]
}
```

## IDE usage

Point Cline / Cursor at the IDE assistant as usual. Choose:

- **`switchyard/agent`** (or `SWITCHYARD_ROUTE_ID`) — escalation / strategy route  
- **Any exposed model id** — direct passthrough to that model’s backend  

```bash
# List models (route + individuals)
curl -H "Authorization: Bearer $API_KEY" http://localhost:8080/v1/models
```

Responses include a `switchyard` object:

```json
{
  "switchyard": {
    "served_by": "weak",
    "latched": false,
    "judge_verdict": "decline",
    "streak": 0,
    "route": "escalation"
  }
}
```

## External switchyard-server

```bash
# Export routes.toml from this config
SWITCHYARD_ROUTES_TOML=./routes.toml

# Or only proxy to an already-running Switchyard binary
SWITCHYARD_ENABLED=true
SWITCHYARD_STRATEGY=external
SWITCHYARD_SERVER_URL=http://127.0.0.1:4000
```

```bash
# Official server (optional)
cargo install --locked switchyard-server
# or: uv tool install --python 3.12 "nemo-switchyard[cli,server]"
switchyard-server --config routes.toml --host 127.0.0.1 --port 4000
```

## Escalation algorithm

Matches NeMo Switchyard escalation router:

1. Call **weak** model, buffer reply  
2. **Judge** scores the completed turn (`escalate` | `decline`)  
3. Consecutive `escalate` count → **streak**  
4. If `streak < confirmations` → serve weak  
5. If `streak >= confirmations` → discard weak, call **strong**, **latch** session  
6. Latched sessions skip the judge and always use strong  
7. Judge errors **fail-open** (serve weak, keep streak)

Session key: `X-Conversation-ID` header, or `user` / `conversation_id` body fields.


## Streaming

IDE assistants call `SwitchyardRouter.chat_completions_stream` when the client
sends `stream: true`:

| Situation | Behavior |
|-----------|----------|
| Direct model id / passthrough / random | Live upstream SSE |
| Capability (after judge picks tier) | Live upstream SSE |
| Escalation already latched | Live strong SSE |
| Escalation unlatched, decline | Buffer weak+judge → synthesize SSE |
| Escalation unlatched, confirmed escalate | Discard weak → live strong SSE |
| `strategy=external` | Live SSE proxy to switchyard-server |

Non-stream `chat_completions` is unchanged (full JSON, including `switchyard` meta).

## Code entry points

- `shared.switchyard.get_switchyard_config()` — load env/file config  
- `shared.switchyard.SwitchyardRouter` — route chat completions  
- `shared.switchyard.EscalationRouter` — in-process escalation  
- `shared.switchyard.export_routes_toml()` — TOML for official server

## Parallel auto-deploy

When apps start with `AUTO_DEPLOY_MODEL=true`, every unique model in this config
is deployed **in parallel** via `shared.deploy.ensure_models_runtime` (deduping
shared backends such as weak+judge on the same URL). Non-IDE apps reuse the same
JSON and expose a **model dropdown** instead of escalation routing.
