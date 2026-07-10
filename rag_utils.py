"""
rag_utils.py
Shared RAG logic for ScholarMind.

This centralizes everything that used to be duplicated between
create_database.py and main.py, so the Streamlit app (and any future
interface) can reuse the same functions instead of copy-pasting.
"""

import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

CHROMA_DIR = "chroma_db"
UPLOAD_DIR = "uploaded_docs"

DEFAULT_MODEL = "gpt-5.4-nano-2026-03-17"  # keep in sync with whatever you were using in main.py

SYSTEM_PROMPT = """You are ScholarMind, a helpful study assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


def get_embeddings():
    return OpenAIEmbeddings()


def get_llm(model: str = DEFAULT_MODEL):
    return ChatOpenAI(model=model)


def load_and_split_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load a single PDF and split it into chunks."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def database_exists(persist_directory: str = CHROMA_DIR) -> bool:
    return os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0


def add_documents_to_store(chunks, persist_directory: str = CHROMA_DIR):
    """Add chunks to the vector store, creating it if it doesn't exist yet."""
    embeddings = get_embeddings()
    if database_exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
    return vectorstore


def get_vectorstore(persist_directory: str = CHROMA_DIR):
    embeddings = get_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def reset_database(persist_directory: str = CHROMA_DIR):
    """Wipe the persisted Chroma store completely (fresh start)."""
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)


def collection_count(persist_directory: str = CHROMA_DIR) -> int:
    """How many chunks are currently indexed. Returns 0 if the store is empty/missing."""
    if not database_exists(persist_directory):
        return 0
    try:
        vectorstore = get_vectorstore(persist_directory)
        return vectorstore._collection.count()
    except Exception:
        return 0


def get_retriever(vectorstore, k: int = 4, fetch_k: int = 10, lambda_mult: float = 0.5):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def answer_question(query: str, retriever, llm):
    """Run retrieval + generation, returning both the answer text and source docs."""
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)
    return response.content, docs


def retrieve_docs(query: str, retriever):
    return retriever.invoke(query)


def stream_answer(query: str, docs, llm):
    """Yield answer text chunks as they're generated (for the web UI's streaming chat)."""
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke({"context": context, "question": query})
    for chunk in llm.stream(final_prompt):
        if chunk.content:
            yield chunk.content


def docs_to_payload(docs):
    """Serialize retrieved documents into plain dicts for JSON responses."""
    payload = []
    for doc in docs:
        payload.append(
            {
                "file": os.path.basename(doc.metadata.get("source", "document")),
                "page": doc.metadata.get("page", "?"),
                "snippet": doc.page_content[:220].replace("\n", " ").strip(),
            }
        )
    return payload


def list_library(persist_directory: str = CHROMA_DIR):
    """Return per-file chunk counts currently indexed, for the sidebar library list."""
    if not database_exists(persist_directory):
        return []
    vectorstore = get_vectorstore(persist_directory)
    try:
        data = vectorstore._collection.get(include=["metadatas"])
    except Exception:
        return []
    counts = {}
    for meta in data.get("metadatas", []):
        name = os.path.basename(meta.get("source", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    return [{"name": name, "chunks": n} for name, n in sorted(counts.items())]


# ------------------------------------------------------------------------
# Session-scoped (in-memory) vector stores — used by the web app so that
# each visitor gets their own private library. Nothing here ever touches
# disk: no persist_directory is passed to Chroma, so the collection lives
# only in server RAM for as long as the process/session is alive.
# ------------------------------------------------------------------------


def new_ephemeral_vectorstore(collection_name: str):
    """A vector store scoped to one session — lives only in memory, never persisted."""
    embeddings = get_embeddings()
    return Chroma(collection_name=collection_name, embedding_function=embeddings)


def add_documents_to_vectorstore(chunks, vectorstore):
    vectorstore.add_documents(chunks)
    return vectorstore


def vectorstore_chunk_count(vectorstore) -> int:
    if vectorstore is None:
        return 0
    try:
        return vectorstore._collection.count()
    except Exception:
        return 0


def list_vectorstore_contents(vectorstore):
    """Same as list_library(), but for an in-memory session vectorstore instead of a path."""
    if vectorstore is None:
        return []
    try:
        data = vectorstore._collection.get(include=["metadatas"])
    except Exception:
        return []
    counts = {}
    for meta in data.get("metadatas", []):
        name = os.path.basename(meta.get("source", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    return [{"name": name, "chunks": n} for name, n in sorted(counts.items())]