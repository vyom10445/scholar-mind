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
