"""
Este módulo define roteadores FastAPI para a criação de embeddings.
"""

from typing import Union
from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse
from ..models.embedding_models import EmbeddingRequest, EmbeddingResponse
from ..services.embedding_services import EmbeddingService


def validate_input(input_text: Union[str, list[str]]) -> None:
    """
    Valida a entrada para a geração de embeddings.
    A entrada deve ser uma string ou uma lista de strings.
    Lança uma exceção HTTPException para entradas inválidas.
    """

    if not input_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A entrada não pode estar vazia.",
        )


def validate_user_header(user_id: str = Header(None)) -> str:
    """
    Valida a presença do cabeçalho 'user-id' na requisição HTTP.
    """

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O header user-id é obrigatório.",
        )

    return user_id


EmbeddingRouter = APIRouter(
    dependencies=[Depends(validate_user_header)],
    responses={status.HTTP_400_BAD_REQUEST: {"description": "Bad Request"}},
)


@EmbeddingRouter.post(
    "/embeddings",
    description="Este endpoint fornece embeddings para a entrada fornecida.",
    operation_id="create_embeddings",
    response_description="Os embeddings gerados",
    response_model=EmbeddingResponse,
    summary="Cria embeddings para a entrada fornecida.",
)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Este endpoint fornece embeddings para a entrada fornecida.
    """

    validate_input(request.input)

    # TODO: Alterar para serviços que usem Chains/Agents/Tools
    embedding_service = EmbeddingService()
    embedding_response = embedding_service.create_embeddings(request.input)

    return JSONResponse(embedding_response.dict(), status_code=status.HTTP_201_CREATED)
