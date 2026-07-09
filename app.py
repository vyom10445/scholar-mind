"""
app.py
ScholarMind — a study companion grounded in your own notes.

Run with:
    streamlit run app.py
"""

import os
import time

import streamlit as st
from dotenv import load_dotenv

from rag_utils import (
    CHROMA_DIR,
    UPLOAD_DIR,
    DEFAULT_MODEL,
    load_and_split_pdf,
    add_documents_to_store,
    get_vectorstore,
    get_retriever,
    get_llm,
    answer_question,
    reset_database,
    collection_count,
    database_exists,
)

load_dotenv()

st.set_page_config(page_title="ScholarMind", page_icon="📖", layout="wide")

# ----------------------------------------------------------------------
# Theme — "library at night": ink navy, aged gold, marginalia sage
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;1,500&family=Source+Sans+3:wght@400;500;600&display=swap');

    :root {
        --ink:        #0F1420;
        --panel:      #171C2C;
        --elevated:   #1F2740;
        --gold:       #C9A227;
        --gold-soft:  #E8CE7A;
        --sage:       #7A9E7E;
        --parchment:  #EDE8DC;
        --muted:      #9AA0B4;
    }

    .stApp { background: var(--ink); color: var(--parchment); font-family: 'Source Sans 3', sans-serif; }

    [data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid rgba(201,162,39,0.15);
    }
    [data-testid="stSidebar"] * { color: var(--parchment) !important; }

    h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--parchment) !important; font-weight: 600 !important; }

    .sm-tagline {
        font-family: 'Fraunces', serif;
        font-style: italic;
        color: var(--gold-soft);
        font-size: 1.05rem;
        margin-top: -0.6rem;
        margin-bottom: 1.4rem;
    }

    .sm-divider { border: none; border-top: 1px solid rgba(201,162,39,0.2); margin: 1rem 0; }

    /* Buttons */
    .stButton>button {
        background: var(--gold);
        color: var(--ink);
        border: none;
        border-radius: 6px;
        font-weight: 600;
        transition: background 0.15s ease;
    }
    .stButton>button:hover { background: var(--gold-soft); color: var(--ink); }

    /* File uploader drop zone */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--elevated);
        border: 1.5px dashed rgba(201,162,39,0.4);
        border-radius: 10px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: var(--elevated);
        border-radius: 10px;
        border-left: 3px solid var(--sage);
        padding: 0.4rem 0.8rem;
    }

    /* Metric-style stat card */
    .sm-stat {
        background: var(--elevated);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        border-left: 3px solid var(--gold);
        margin-bottom: 0.8rem;
    }
    .sm-stat .n { font-family: 'Fraunces', serif; font-size: 1.6rem; color: var(--gold-soft); }
    .sm-stat .l { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }

    /* Source citation card, styled like a manuscript margin note */
    .sm-source {
        border-left: 2px solid var(--gold);
        padding-left: 0.7rem;
        margin: 0.4rem 0;
        color: var(--muted);
        font-size: 0.88rem;
    }
    .sm-source b { color: var(--gold-soft); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown("# 📖 ScholarMind")
st.markdown('<div class="sm-tagline">Your study companion, grounded entirely in your own notes.</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Sidebar — library management
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Your Library")

    chunk_count = collection_count()
    st.markdown(
        f'<div class="sm-stat"><div class="n">{chunk_count}</div>'
        f'<div class="l">indexed chunks</div></div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Add PDFs — textbooks, lecture notes, papers",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("➕ Add to Library", use_container_width=True, disabled=not uploaded_files):
        progress = st.progress(0.0, text="Starting…")
        total_chunks = 0
        for i, file in enumerate(uploaded_files):
            progress.progress((i) / len(uploaded_files), text=f"Reading {file.name}…")
            save_path = os.path.join(UPLOAD_DIR, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

            chunks = load_and_split_pdf(save_path)
            add_documents_to_store(chunks)
            total_chunks += len(chunks)

        progress.progress(1.0, text="Done!")
        time.sleep(0.4)
        progress.empty()
        st.success(f"Added {len(uploaded_files)} file(s) — {total_chunks} chunks indexed.")
        st.rerun()

    st.markdown('<hr class="sm-divider">', unsafe_allow_html=True)

    with st.expander("⚙️ Retrieval & model settings"):
        model_name = st.text_input("Model", value=DEFAULT_MODEL)
        k = st.slider("Results to retrieve (k)", 2, 10, 4)
        fetch_k = st.slider("Candidates before re-ranking (fetch_k)", k, 20, 10)
        lambda_mult = st.slider("Diversity (lambda_mult)", 0.0, 1.0, 0.5)

    with st.expander("🗑️ Danger zone"):
        st.caption("Wipes every indexed document. Cannot be undone.")
        if not st.session_state.confirm_reset:
            if st.button("Reset database", use_container_width=True):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            st.warning("Are you sure? This deletes all indexed chunks.")
            c1, c2 = st.columns(2)
            if c1.button("Yes, delete", use_container_width=True):
                reset_database()
                st.session_state.messages = []
                st.session_state.confirm_reset = False
                st.success("Database cleared.")
                st.rerun()
            if c2.button("Cancel", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()

# ----------------------------------------------------------------------
# Main — chat
# ----------------------------------------------------------------------
if not database_exists():
    st.info("Your library is empty. Upload a PDF from the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "📖"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for doc in msg["sources"]:
                        name = os.path.basename(doc.metadata.get("source", "document"))
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content[:180].replace("\n", " ").strip()
                        st.markdown(
                            f'<div class="sm-source"><b>{name}</b> — page {page}<br>{snippet}…</div>',
                            unsafe_allow_html=True,
                        )

    query = st.chat_input("Ask something about your notes…")
    if query:
        st.session_state.messages.append({"role": "user", "content": query, "sources": None})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="📖"):
            with st.spinner("Thinking…"):
                vectorstore = get_vectorstore()
                retriever = get_retriever(vectorstore, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
                llm = get_llm(model=model_name)
                answer, docs = answer_question(query, retriever, llm)
            st.markdown(answer)
            if docs:
                with st.expander(f"Sources ({len(docs)})"):
                    for doc in docs:
                        name = os.path.basename(doc.metadata.get("source", "document"))
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content[:180].replace("\n", " ").strip()
                        st.markdown(
                            f'<div class="sm-source"><b>{name}</b> — page {page}<br>{snippet}…</div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": docs})
