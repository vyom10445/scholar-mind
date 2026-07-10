"""
server.py
ScholarMind backend — FastAPI app serving the web UI and RAG endpoints.

Every visitor gets their own private, in-memory library, identified by a
session cookie. Uploaded PDFs are read into a temp file just long enough
to extract text, then discarded. Nothing — not the PDFs, not the vector
store — is ever written into the project folder, so this is safe to
deploy for multiple simultaneous users.

Run with:
    uvicorn server:app --reload

Note: this uses a single process's memory to hold sessions. That's fine
for `uvicorn server:app` (one worker) on a typical small deployment. If
you ever scale to multiple worker processes/instances behind a load
balancer, sessions would need to move to a shared store (e.g. Redis, or
a per-session Chroma directory on a shared volume) since each worker
would otherwise have its own separate memory.
"""

import asyncio
import json
import os
import tempfile
import time
import uuid
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rag_utils import (
    DEFAULT_MODEL,
    load_and_split_pdf,
    new_ephemeral_vectorstore,
    add_documents_to_vectorstore,
    vectorstore_chunk_count,
    list_vectorstore_contents,
    get_retriever,
    get_llm,
    retrieve_docs,
    stream_answer,
    docs_to_payload,
)

load_dotenv()

app = FastAPI(title="ScholarMind")

SESSION_COOKIE = "sm_session"
SESSION_TTL_SECONDS = 2 * 60 * 60  # sessions idle longer than this get purged

# session_id -> {"vectorstore": Chroma | None, "last_active": float}
SESSIONS: dict = {}


def touch_session(session_id: str):
    """Get (or create) a session's slot and mark it as just used."""
    session = SESSIONS.setdefault(session_id, {"vectorstore": None, "last_active": time.time()})
    session["last_active"] = time.time()
    return session


def purge_expired_sessions():
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items() if now - s["last_active"] > SESSION_TTL_SECONDS]
    for sid in expired:
        vs = SESSIONS[sid].get("vectorstore")
        if vs is not None:
            try:
                vs.delete_collection()
            except Exception:
                pass
        del SESSIONS[sid]


@app.on_event("startup")
async def start_cleanup_loop():
    async def loop():
        while True:
            await asyncio.sleep(600)
            purge_expired_sessions()
    asyncio.create_task(loop())


def get_session_id(request: Request) -> tuple[str, bool]:
    """Returns (session_id, is_new). Reuses the cookie if present and still valid."""
    session_id = request.cookies.get(SESSION_COOKIE)
    is_new = session_id is None or session_id not in SESSIONS
    if session_id is None:
        session_id = str(uuid.uuid4())
    return session_id, is_new


def attach_cookie(response, session_id: str):
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
    )
    return response


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/library")
def get_library(request: Request):
    session_id, _ = get_session_id(request)
    session = touch_session(session_id)
    vs = session["vectorstore"]

    payload = {
        "files": list_vectorstore_contents(vs),
        "total_chunks": vectorstore_chunk_count(vs),
        "has_documents": vectorstore_chunk_count(vs) > 0,
    }
    response = JSONResponse(payload)
    return attach_cookie(response, session_id)


@app.post("/api/upload")
async def upload(request: Request, files: List[UploadFile] = File(...)):
    session_id, _ = get_session_id(request)
    session = touch_session(session_id)

    if session["vectorstore"] is None:
        session["vectorstore"] = new_ephemeral_vectorstore(collection_name=f"sm_{session_id}")
    vectorstore = session["vectorstore"]

    added = []
    total_chunks = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for file in files:
            tmp_path = os.path.join(tmp_dir, file.filename)
            with open(tmp_path, "wb") as f:
                f.write(await file.read())

            chunks = load_and_split_pdf(tmp_path)
            add_documents_to_vectorstore(chunks, vectorstore)
            added.append({"name": file.filename, "chunks": len(chunks)})
            total_chunks += len(chunks)
        # tmp_dir (and every file in it) is deleted automatically here

    payload = {
        "added": added,
        "total_chunks_added": total_chunks,
        "library": list_vectorstore_contents(vectorstore),
    }
    response = JSONResponse(payload)
    return attach_cookie(response, session_id)


@app.post("/api/reset")
def reset(request: Request):
    session_id, _ = get_session_id(request)
    session = touch_session(session_id)

    if session["vectorstore"] is not None:
        try:
            session["vectorstore"].delete_collection()
        except Exception:
            pass
        session["vectorstore"] = None

    response = JSONResponse({"status": "ok"})
    return attach_cookie(response, session_id)


@app.post("/api/chat")
async def chat(request: Request, payload: dict):
    session_id, _ = get_session_id(request)
    session = touch_session(session_id)
    vectorstore = session["vectorstore"]

    query = (payload.get("query") or "").strip()
    k = int(payload.get("k", 4))
    fetch_k = int(payload.get("fetch_k", 10))
    lambda_mult = float(payload.get("lambda_mult", 0.5))
    model = payload.get("model") or DEFAULT_MODEL

    if not query or vectorstore is None or vectorstore_chunk_count(vectorstore) == 0:
        def empty_gen():
            yield json.dumps({"type": "sources", "data": []}) + "\n"
            msg = "Your library is empty — upload a PDF first." if not vectorstore_chunk_count(vectorstore) else "Ask me something!"
            yield json.dumps({"type": "token", "data": msg}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        response = StreamingResponse(empty_gen(), media_type="text/plain")
        return attach_cookie(response, session_id)

    retriever = get_retriever(vectorstore, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    docs = retrieve_docs(query, retriever)
    llm = get_llm(model=model)

    def generate():
        yield json.dumps({"type": "sources", "data": docs_to_payload(docs)}) + "\n"
        try:
            for chunk in stream_answer(query, docs, llm):
                yield json.dumps({"type": "token", "data": chunk}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    response = StreamingResponse(generate(), media_type="text/plain")
    return attach_cookie(response, session_id)