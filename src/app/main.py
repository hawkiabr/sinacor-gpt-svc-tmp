"""
Módulo para criar e configurar uma aplicação FastAPI.
Este módulo define a aplicação FastAPI, inclui os roteadores necessários e cria o esquema OpenAPI.
"""

from typing import Any, Dict
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from .routers import chat_routers, embedding_routers, health_routers


def create_app():
    """
    Cria e configura a aplicação FastAPI.
    """

    app = FastAPI()  # FastAPI app
    app.include_router(chat_routers.ChatRouter, prefix="/api", tags=["chat"])
    app.include_router(embedding_routers.EmbeddingRouter, prefix="/api", tags=["embeddings"])
    app.include_router(health_routers.HealthRouter, tags=["health"])
    app.openapi = lambda: create_openapi(app)
    return app


def create_openapi(app: FastAPI) -> Dict[str, Any]:
    """
    Cria o esquema OpenAPI Specification (OAS 3.1) da API.
    """

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="SinacorGPT | REST API",
        version="v1.0.0",
        routes=app.routes,
        summary="Referência da API REST do SinacorGPT",
        description=(
            "Crie interações, conclusões para mensagens de chat ou obtenha "
            "uma representação vetorial (embeddings) de uma determinada entrada que pode "
            "ser consumida por modelos ChatGPT do Azure OpenAI, utilizando dados sobre o Sinacor."
        ),
    )

    openapi_schema["info"]["x-logo"] = {
        "url": "https://www.hawkia.com.br/assets/images/hwk-img-logo-horizontal-neg-escuro.svg"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


fastapi_app = create_app()
