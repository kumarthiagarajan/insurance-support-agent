import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY in your environment or .env file before running.")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from graph import build_graph

STATIC_DIR = Path(__file__).parent / "static"
NO_REPLY_FALLBACK = (
    "No response was returned for this step. Please try rephrasing your question, "
    "or ask to speak with a representative."
)

app = FastAPI(title="Insurance Support Agent API")
_graph = build_graph()
_sessions: dict[str, dict] = {}


class StartSessionRequest(BaseModel):
    customer_id: str


class StartSessionResponse(BaseModel):
    session_id: str
    customer_id: str


class MessageRequest(BaseModel):
    message: str


class SpecialistReply(BaseModel):
    speaker: str
    content: str


class MessageResponse(BaseModel):
    replies: list[SpecialistReply]


def _new_state(customer_id: str) -> dict:
    return {
        "messages": [],
        "customer_id": customer_id,
        "next": "",
        "handled": [],
        "iterations": 0,
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    customer_id = req.customer_id.strip()
    if not customer_id:
        raise HTTPException(status_code=400, detail="customer_id is required")

    session_id = str(uuid.uuid4())
    _sessions[session_id] = _new_state(customer_id)
    return StartSessionResponse(session_id=session_id, customer_id=customer_id)


@app.post("/api/session/{session_id}/message", response_model=MessageResponse)
def send_message(session_id: str, req: MessageRequest):
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    prev_len = len(state["messages"])
    state["messages"].append(HumanMessage(content=req.message))
    state["handled"] = []
    state["iterations"] = 0

    state = _graph.invoke(state)
    _sessions[session_id] = state

    replies = [
        SpecialistReply(
            speaker=getattr(m, "name", None) or "assistant",
            content=m.content or NO_REPLY_FALLBACK,
        )
        for m in state["messages"][prev_len + 1 :]
        if m.type == "ai"
    ]
    return MessageResponse(replies=replies)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
