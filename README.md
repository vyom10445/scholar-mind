# ScholarMind

**An AI study companion that answers questions strictly from your own notes.**

ScholarMind is a Retrieval-Augmented Generation (RAG) application. Users upload their own study materials — lecture slides, textbooks, research papers — and ask questions through a chat interface. Every answer is grounded in the uploaded material; if the answer isn't present in the source documents, the system says so instead of generating unsupported content.

---

## Features

- **Document upload** — supports PDF ingestion for textbooks, lecture notes, and papers
- **Streaming chat interface** — responses are generated and rendered token by token
- **Source attribution** — every answer references the specific file and page it was derived from
- **Session isolation** — each user receives a private, in-memory document library; no data is shared between users or persisted to disk
- **Configurable retrieval** — chunk count, candidate pool size, and result diversity (MMR) are adjustable at runtime
- **Session reset** — users can clear their library at any time

---

## Architecture

1. **Ingestion** — uploaded PDFs are parsed and split into overlapping text chunks
2. **Embedding** — each chunk is converted into a vector representation using an OpenAI embedding model
3. **Indexing** — vectors are stored in a Chroma vector database, scoped to the user's session
4. **Retrieval** — the user's query is embedded and matched against the most relevant chunks using Maximal Marginal Relevance (MMR) search
5. **Generation** — retrieved chunks are passed to an LLM with an instruction to answer only from the provided context, and the response is streamed back to the client

```
PDF → chunking → embeddings → Chroma (in-memory, per session) → retrieval → LLM → streamed response
```

---

## Tech stack

| Layer              | Technology                                      |
|---------------------|--------------------------------------------------|
| Backend             | FastAPI (Python)                                 |
| RAG orchestration   | LangChain                                        |
| Vector store        | Chroma (in-memory, session-scoped)               |
| Embeddings / LLM    | OpenAI API                                       |
| Frontend            | HTML / CSS / JavaScript (no framework)           |
| PDF parsing         | PyPDF                                            |

---

## Project structure

```
ScholarMind/
├── server.py            # FastAPI app — routes, session management, streaming chat
├── rag_utils.py          # Shared RAG logic: load, chunk, embed, retrieve, generate
├── static/
│   ├── index.html         # Application shell
│   ├── style.css            # Design system and layout
│   └── script.js              # Upload handling, chat streaming, UI logic
├── create_database.py    # Standalone CLI script for building a persistent local library
├── main.py                # Standalone CLI chat script (terminal-based RAG)
├── requirements.txt
└── .env                   # API keys (not committed)
```

---

## Getting started

### 1. Install dependencies
```bash
pip install fastapi uvicorn python-multipart langchain langchain-community langchain-openai langchain-text-splitters chromadb pypdf python-dotenv --break-system-packages
```

### 2. Configure environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
```

### 3. Run the application
```bash
uvicorn server:app --reload
```

Open `http://127.0.0.1:8000` in a browser, upload a PDF, and begin querying the document.

---

## Privacy and deployment notes

- Uploaded files are written to a temporary directory only for the duration of text extraction, then deleted immediately.
- Each user's document library exists only in server memory for the duration of their session and expires automatically after a period of inactivity.
- No document content is persisted to disk or shared across sessions, making the application suitable for public, multi-user deployment.

---

## Roadmap

- Support for additional file formats (`.docx`, `.pptx`, `.txt`)
- Multi-turn conversational memory
- Exportable chat history and study notes
- Optional persistent libraries for returning users

---
