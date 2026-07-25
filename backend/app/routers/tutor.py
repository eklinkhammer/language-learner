from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services import llm
from app.services.llm import LLMServiceError

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Conversational language tutor powered by claude -p."""
    try:
        result = await llm.chat(
            message=request.message,
            language=request.language,
            history=[m.model_dump() for m in request.history],
        )
    except LLMServiceError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return ChatResponse(
        reply=result["reply"],
        suggestions=result["suggestions"],
    )
