from fastapi import APIRouter

from pydantic import BaseModel

from app.services.rag_service import (
    chat_with_rag
)

router = APIRouter()


class ChatRequest(BaseModel):

    question: str


@router.post("/chat")
def chat(request: ChatRequest):

    result = chat_with_rag(
        request.question
    )

    return result