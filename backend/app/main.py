from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes_chatbot import router as chatbot_router

app = FastAPI(title="Teacher Support Chatbot")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(chatbot_router)


@app.get("/")
def root():
	return {"message": "Chatbot service running"}
