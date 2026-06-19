"""Premium Streamlit frontend for the AI Customer Support Bot."""
import os
import requests
import streamlit as st

API = os.getenv("BACKEND_URL", "http://localhost:8000")

# Branding
APP_LOGO = "🛟"
APP_NAME = "SupportAI"
APP_TAGLINE = "AI agent for customer support"

st.set_page_config(
    page_title=f"{APP_NAME} — AI Customer Support",
    page_icon=APP_LOGO,
    layout="centered",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

    /* Soft warm gradient background */
    .stApp {
        background: linear-gradient(180deg, #FAF8F5 0%, #F5F3FF 100%);
        background-attachment: fixed;
    }
    #MainMenu, footer, header { visibility: hidden; }
    h1, h2, h3 { color: #111827; font-weight: 700; letter-spacing: -0.02em; }

    /* Brand header */
    .brand-wrap {
        display: flex; align-items: center; gap: 14px;
        padding: 8px 0;
    }
    .brand-logo {
        font-size: 1.8rem; line-height: 1;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        width: 48px; height: 48px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 14px;
        box-shadow: 0 8px 24px -4px rgba(99, 102, 241, 0.4);
    }
    .brand-name {
        font-size: 1.25rem; font-weight: 700; color: #111827;
        letter-spacing: -0.02em; line-height: 1.2;
    }
    .brand-tagline {
        font-size: 0.8rem; color: #6B7280; margin-top: 2px;
    }

    /* Primary buttons — gradient */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white; border: none;
        border-radius: 12px; font-weight: 600; padding: 0.7rem 1.4rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px -2px rgba(99, 102, 241, 0.4);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -2px rgba(99, 102, 241, 0.5);
    }

    /* Secondary buttons */
    .stButton > button:not([kind="primary"]) {
        background-color: #FFFFFF; color: #111827;
        border: 1px solid #E5E7EB; border-radius: 12px;
        font-weight: 500; padding: 0.7rem 1rem;
        transition: all 0.15s ease;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: #8B5CF6; color: #6366F1;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px -2px rgba(139, 92, 246, 0.15);
    }

    /* Chat input */
    .stChatInput textarea {
        border-radius: 16px !important;
        border: 1.5px solid #E5E7EB !important;
        background-color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.9rem 1rem !important;
    }
    .stChatInput textarea:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.12) !important;
    }

    /* Welcome */
    .welcome-title {
        font-size: 2.4rem; font-weight: 700; color: #111827;
        text-align: center; margin-top: 1.5rem; margin-bottom: 0.5rem;
        letter-spacing: -0.035em;
    }
    .welcome-subtitle {
        font-size: 1rem; color: #6B7280;
        text-align: center; margin-bottom: 2rem;
    }

    /* Status pills */
    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 12px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600;
    }
    .pill-online {
        background-color: #ECFDF5; color: #047857;
        border: 1px solid #A7F3D0;
    }
    .pill-offline {
        background-color: #FEE2E2; color: #B91C1C;
        border: 1px solid #FECACA;
    }
    .status-dot {
        width: 7px; height: 7px; border-radius: 50%;
    }
    .dot-green { background-color: #10B981; animation: pulse 2s infinite; }
    .dot-red { background-color: #EF4444; }
    @keyframes pulse { 0%,100% {opacity:1;} 50% {opacity:0.4;} }

    /* Divider color */
    hr { border-color: #E5E7EB; }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_response(text: str) -> str:
    """Strip [Source N] markers and escape $ to prevent LaTeX rendering."""
    import re
    text = re.sub(r"\s*\[Source \d+\]", "", text)
    text = text.replace("$", "\\$")
    return text.strip()


def clean_user_input(text: str) -> str:
    """Just escape $ for user messages."""
    return text.replace("$", "\\$")


# ============================================================
# STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-session"
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# BACKEND CALL
# ============================================================
def ask_backend(question):
    try:
        r = requests.post(
            f"{API}/ask",
            json={"question": question, "session_id": st.session_state.session_id},
            timeout=120,
        )
        if not r.ok:
            try:
                err = r.json().get("detail", r.text)
            except Exception:
                err = r.text
            st.session_state.messages.append({
                "role": "assistant", "content": f"⚠️ {err}",
            })
        else:
            data = r.json()
            st.session_state.messages.append({
                "role": "assistant", "content": data["answer"],
            })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant", "content": f"⚠️ Connection error: {e}",
        })


# ============================================================
# TOP HEADER
# ============================================================
backend_ok = False
backend_info = {}
try:
    backend_info = requests.get(f"{API}/health", timeout=10).json()
    backend_ok = True
except Exception:
    backend_ok = False

col_brand, col_status = st.columns([3, 2])
with col_brand:
    st.markdown(
        f"""
        <div class="brand-wrap">
            <div class="brand-logo">{APP_LOGO}</div>
            <div>
                <div class="brand-name">{APP_NAME}</div>
                <div class="brand-tagline">{APP_TAGLINE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_status:
    if backend_ok:
        st.markdown(
            f"""
            <div style="text-align: right; margin-top: 14px;">
                <span class="status-pill pill-online"><span class="status-dot dot-green"></span>Online</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="text-align: right; margin-top: 14px;">
                <span class="status-pill pill-offline"><span class="status-dot dot-red"></span>Backend offline</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

if not backend_ok:
    st.error("Cannot reach the backend API. Make sure `uvicorn` is running on port 8000.")
    st.stop()

# ============================================================
# DOC UPLOAD + NEW CHAT
# ============================================================
with st.expander("📁 Knowledge Base · Upload Documents", expanded=(backend_info.get("doc_count", 0) == 0)):
    files = st.file_uploader(
        "Drop PDF, TXT, or CSV files here",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv"],
    )
    col_up, col_clear = st.columns([2, 1])
    with col_up:
        if st.button("⬆️ Upload & Index", type="primary", disabled=not files, use_container_width=True):
            with st.spinner("Indexing documents..."):
                payload = [
                    ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                    for f in files
                ]
                try:
                    r = requests.post(f"{API}/upload", files=payload, timeout=300)
                    if r.ok:
                        data = r.json()
                        st.success(
                            f"✨ Indexed {data['chunks_added']} chunks from {len(data['files'])} files"
                        )
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {r.text}")
                except Exception as e:
                    st.error(f"Upload error: {e}")
    with col_clear:
        if st.button("＋ New Chat", use_container_width=True):
            try:
                requests.post(
                    f"{API}/clear",
                    json={"session_id": st.session_state.session_id},
                    timeout=10,
                )
            except Exception:
                pass
            st.session_state.messages = []
            st.rerun()

# ============================================================
# MAIN
# ============================================================
if not st.session_state.messages:
    st.markdown('<div class="welcome-title">How can I help you today?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="welcome-subtitle">Ask anything about your uploaded documents</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    suggestions = [
        ("📦 Track Order", "How can I track my order?"),
        ("↩️ Returns & Refunds", "What is your return and refund policy?"),
        ("💳 Billing Questions", "How does billing work?"),
        ("🛠️ Product Support", "What support options do you offer?"),
    ]
    for i, (label, prompt) in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(label, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_question = prompt
                st.rerun()

# Render conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(clean_response(msg["content"]))
        else:
            st.markdown(clean_user_input(msg["content"]))

# Handle suggested prompt
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(clean_user_input(q))
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            ask_backend(q)
    st.rerun()

# Chat input
if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(clean_user_input(question))
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            ask_backend(question)
    st.rerun()