from fastapi import APIRouter
from pydantic import BaseModel

from app.services.rag_service import rag_service
from app.services.chat_history_service import (
    chat_history_service
)

router = APIRouter()


class ChatRequest(BaseModel):

    session_id: str
    question: str


@router.post("/chat")
def chat(request: ChatRequest):

    chat_history_service.save_message(
        request.session_id,
        "user",
        request.question
    )

    result = (
        rag_service.chat_with_rag(
            question=request.question
        )
    )

    chat_history_service.save_message(
        request.session_id,
        "assistant",
        result["answer"]
    )

    return result


@router.get(
    "/history/{session_id}"
)
def get_history(
    session_id: str
):

    return (
        chat_history_service
        .get_history(session_id)
    )


@router.get("/sessions")
def get_sessions():

    return (
        chat_history_service
        .get_sessions()
    )