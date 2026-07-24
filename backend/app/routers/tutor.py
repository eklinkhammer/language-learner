from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Conversational language tutor powered by claude -p."""
    # TODO: Implement claude -p subprocess call via services/llm.py
    return ChatResponse(
        reply=f"(stub) ¡Hola! I'm your {request.language} tutor. How can I help you practice today?",
        suggestions=[
            "¿Cómo estás?",
            "Me llamo...",
            "¿Dónde está...?",
        ],
    )
