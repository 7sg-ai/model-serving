"""
HuggingFace Model with Streamlit Chat Interface (vLLM backend)
Supports optional login, chat history, and multi-model dropdown.
Inference via OpenAI-compatible vLLM (one or more models in parallel).
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory
from shared.backends import require_backend_urls
from shared.multinode import BackendPool, ClusterInfo, get_multinode_config
from shared.tools import get_web_access_tool, get_web_enricher
from shared.model_catalog import deploy_app_models, load_app_models

os.environ.setdefault("LOAD_MODEL_WEIGHTS", "false")

MN_CFG = get_multinode_config(default_port=8501, app_name="hf-streamlit-chat")
CLUSTER = ClusterInfo(MN_CFG, app_name="hf-streamlit-chat")
WEB = get_web_access_tool("hf-streamlit-chat")
ENRICH = get_web_enricher("hf-streamlit-chat")

DEFAULT_MODEL_NAME = os.getenv(
    "HF_MODEL_NAME", os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
)
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
VLLM_API_KEY = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
REQUEST_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "300"))

default_backend = (MN_CFG.backend_urls[0] if MN_CFG.backend_urls else "")
SY_CFG = load_app_models(
    default_model_id=DEFAULT_MODEL_NAME,
    default_backend_url=default_backend,
    owned_by="hf-streamlit-chat",
    reload=True,
)
CATALOG, _HANDLES = deploy_app_models(
    "vllm",
    SY_CFG,
    app_name="hf-streamlit-chat",
    api_key=VLLM_API_KEY,
    default_backend_urls=MN_CFG.backend_urls or CLUSTER.backend_pool.urls,
)
backends = require_backend_urls(
    CATALOG.all_backend_urls() or MN_CFG.backend_urls or CLUSTER.backend_pool.urls
)
CLUSTER.backend_pool = BackendPool(backends, strategy=MN_CFG.backend_strategy)
MN_CFG.backend_urls = backends
MODEL_NAME = CATALOG.default_id()
print(
    f"Models: {[m.id for m in CATALOG.models]} multi={CATALOG.multi} "
    f"backends={backends}"
)

auth_manager = AuthManager()
chat_history = ChatHistory()


def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_token" not in st.session_state:
        st.session_state.session_token = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = CATALOG.default_id()


def login_page():
    st.title("Login / Register")
    tab1, tab2, tab3 = st.tabs(["Login", "Register", "Continue as Guest"])

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            session_token = auth_manager.login(username, password)
            if session_token:
                st.session_state.session_token = session_token
                st.session_state.user_id = auth_manager.validate_session(session_token)
                st.session_state.username = username
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        st.subheader("Register")
        new_username = st.text_input("Username", key="register_username")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input(
            "Confirm Password", type="password", key="confirm_password"
        )
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters")
            elif auth_manager.register_user(new_username, new_password):
                st.success("Registration successful! Please login.")
            else:
                st.error("Username already exists")

    with tab3:
        st.subheader("Continue as Guest")
        st.info("Your chat history will not be saved")
        if st.button("Continue as Guest"):
            st.session_state.authenticated = True
            st.session_state.username = "Guest"
            st.rerun()


def generate_response(prompt, messages, web_context: str = "", model_id: str = ""):
    model_messages = []
    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "assistant", "system") and content:
            model_messages.append({"role": role, "content": content})
    if model_messages and model_messages[-1].get("role") == "user":
        model_messages = model_messages[:-1]
    user_content = prompt
    if web_context:
        user_content = (
            f"{web_context}\n\n"
            f"User question (may reference the URL(s) above):\n{prompt}"
        )
    model_messages.append({"role": "user", "content": user_content})
    try:
        text, _data = CATALOG.chat(
            model_messages,
            model=model_id or st.session_state.get("selected_model"),
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return text or "(empty response)"
    except Exception as e:
        return f"Error: {e}"


def chat_page():
    st.title("HuggingFace Chat Interface")

    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.username}**")
        labels, label_to_id = CATALOG.streamlit_options()
        current = st.session_state.get("selected_model") or CATALOG.default_id()
        # Map current id to label
        id_to_label = {v: k for k, v in label_to_id.items()}
        default_label = id_to_label.get(current, labels[0] if labels else current)
        try:
            idx = labels.index(default_label)
        except ValueError:
            idx = 0
        chosen = st.selectbox("Model", labels, index=idx, key="model_selectbox")
        st.session_state.selected_model = label_to_id.get(chosen, CATALOG.default_id())
        st.caption(f"Active: `{st.session_state.selected_model}`")
        if CATALOG.multi:
            st.caption(f"{len(CATALOG.choices_for_ui())} models deployed in parallel")

        if st.button("Logout"):
            if st.session_state.session_token:
                auth_manager.logout(st.session_state.session_token)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()

        if st.button("New Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

        if st.session_state.user_id:
            st.subheader("Previous Conversations")
            conversations = chat_history.get_user_conversations(
                st.session_state.user_id, "hf-streamlit-chat"
            )
            for conv in conversations[:10]:
                if st.button(
                    f"Conv {conv['id']} - {conv['updated_at'][:16]}",
                    key=f"conv_{conv['id']}",
                ):
                    st.session_state.conversation_id = conv["id"]
                    messages = chat_history.get_conversation_history(conv["id"])
                    st.session_state.messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in messages
                    ]
                    st.rerun()

    if st.session_state.conversation_id is None and st.session_state.user_id:
        st.session_state.conversation_id = chat_history.create_conversation(
            st.session_state.user_id, "hf-streamlit-chat"
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.conversation_id:
            chat_history.add_message(st.session_state.conversation_id, "user", prompt)

        session_key = str(
            st.session_state.conversation_id
            or st.session_state.user_id
            or st.session_state.username
            or "guest"
        )
        enrichment = ENRICH.enrich_user_text(prompt, session_id=session_key)
        web_context = enrichment.context
        if enrichment.search_results or enrichment.search_error:
            with st.expander(
                f"Web search: {enrichment.query or '(none)'} "
                f"({len(enrichment.search_results)} hit(s))"
            ):
                if enrichment.search_error and not enrichment.search_results:
                    st.warning(enrichment.search_error)
                for row in enrichment.ui_search_rows():
                    title = row.get("title") or ""
                    url = row.get("url") or ""
                    snip = row.get("snippet") or ""
                    st.markdown("- **" + title + "** — `" + url + "`\n  " + snip)

        if enrichment.fetch_results:
            with st.expander(
                f"Fetched {len(enrichment.fetch_results)} URL(s) (full content logged)"
            ):
                for wr in enrichment.fetch_results:
                    st.markdown(
                        f"- `{wr.url}` — "
                        f"{'OK' if wr.ok else 'FAIL'} "
                        f"status={wr.status_code} "
                        f"log=`{wr.log_path}`"
                    )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prior = st.session_state.messages[:-1]
                response = generate_response(
                    prompt,
                    prior + [{"role": "user", "content": prompt}],
                    web_context=web_context,
                    model_id=st.session_state.selected_model,
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        if st.session_state.conversation_id:
            chat_history.add_message(
                st.session_state.conversation_id, "assistant", response
            )

        st.rerun()


def main():
    st.set_page_config(
        page_title="HuggingFace Chat Interface",
        page_icon="🤗",
        layout="wide",
    )
    init_session_state()
    if not st.session_state.authenticated:
        login_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()
