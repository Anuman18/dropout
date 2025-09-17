from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..chatbot.chatbot import Chatbot

router = APIRouter(prefix="/chat", tags=["chatbot"])


class ChatRequest(BaseModel):
	message: str = Field(..., example="Suggest interventions for a struggling student")
	risk_tier: Optional[str] = Field(None, example="red")


# Initialize chatbot once
BOT = Chatbot(Path(__file__).resolve().parents[1] / "chatbot" / "intents.json")


@router.post("")
def chat_endpoint(req: ChatRequest) -> Dict[str, Any]:
	response = BOT.respond(req.message, risk_tier=req.risk_tier)
	return {"response": response}
