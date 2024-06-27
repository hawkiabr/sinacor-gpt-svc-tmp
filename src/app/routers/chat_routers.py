from typing import List
from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse, Response
from ..models.chat_models import ChatMessage, ChatRequest, ChatResponse, ChatRole
from ..services.chat_services import ChatService


def validate_messages(messages: List[ChatMessage]) -> None:
    """
    Valida uma lista de mensagens. Cada mensagem deve ter um 'content' e um 'role'.
    'role' deve ser 'system', 'user' ou 'assistant'.
    Lança uma exceção HTTPException para mensagens inválidas.
    """

    if not messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A lista de mensagens não pode estar vazia.",
        )

    for i, message in enumerate(messages):
        if not message.content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"O atributo 'content' da mensagem no índice {i} não pode estar vazio.",
            )

        if not message.role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"O atributo 'role' da mensagem no índice {i} não pode estar vazio.",
            )

        if message.role not in ChatRole:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"O atributo 'role' da mensagem no índice {i} deve ser um dos seguintes: {', '.join([role.value for role in ChatRole])}",
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


ChatRouter = APIRouter(
    dependencies=[Depends(validate_user_header)],
    responses={status.HTTP_400_BAD_REQUEST: {"description": "Bad Request"}},
)


@ChatRouter.post(
    "/chat",
    description="Este endpoint fornece uma interação de chat com base na solicitação de chat fornecida.",
    operation_id="create_chat_response",
    response_description="A interação de chat gerada",
    response_model=ChatResponse,
    summary="Obtém uma interação de chat.",
)
async def create_chat_response(request: ChatRequest) -> Response:
    """
    Este endpoint fornece uma interação de chat com base na solicitação de chat fornecida.
    Retorna a resposta como StreamingResponse ou JSONResponse,
    com base no atributo stream da solicitação.
    """

    validate_messages(request.messages)

    # TODO: Alterar para serviços que usem Chains/Agents/Tools
    chat_service = ChatService()
    chat_response = chat_service.get_chat_completion(request.messages)

    # TODO: Refatorar entrega de streaming (adequar o futuro front-end)
    # Função geradora para a resposta como streaming
    async def response_generator():
        for choice in chat_response.choices:
            yield " ".join(choice.message.content.split())

    if request.stream:
        return StreamingResponse(
            response_generator(),
            status_code=status.HTTP_200_OK,
        )

    return JSONResponse(
        chat_response.dict(),
        status_code=status.HTTP_200_OK,
    )


@ChatRouter.post(
    "/chat/completion",
    description="Este endpoint fornece uma conclusão de chat com base na solicitação de chat fornecida.",
    operation_id="create_chat_completion",
    response_description="A conclusão de chat gerada",
    response_model=ChatResponse,
    summary="Obtém uma conclusão de chat.",
)
async def create_chat_completion(request: ChatRequest) -> Response:
    """
    Este endpoint fornece uma conclusão de chat com base na solicitação de chat fornecida.
    Retorna a resposta como StreamingResponse ou JSONResponse,
    com base no atributo stream da solicitação.
    """

    validate_messages(request.messages)

    # TODO: Alterar para serviços que usem Chains/Agents/Tools
    chat_service = ChatService()
    chat_response = chat_service.get_chat_completion(request.messages)

    # TODO: Refatorar entrega de streaming (adequar o futuro front-end)
    # Função geradora para a resposta como streaming
    async def response_generator():
        for choice in chat_response.choices:
            yield " ".join(choice.message.content.split())

    if request.stream:
        return StreamingResponse(
            response_generator(),
            status_code=status.HTTP_200_OK,
        )

    return JSONResponse(
        chat_response.dict(),
        status_code=status.HTTP_200_OK,
    )
