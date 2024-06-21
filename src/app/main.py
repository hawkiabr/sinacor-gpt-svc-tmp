from fastapi import FastAPI

from .routers import chat_routers, health_routers

fastapi_app = FastAPI()  # FastAPI app
fastapi_app.include_router(chat_routers.ChatRouter, prefix="/api", tags=["chat"])
fastapi_app.include_router(health_routers.HealthRouter, tags=["health"])
