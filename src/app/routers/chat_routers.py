"""
Este módulo define roteadores FastAPI para interações e conclusões de chat.
"""

from typing import List
from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse, Response
from ..models.chat_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ApiChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
)

from ..services.chat_services import ChatService


def validate_messages(messages: List[ApiChatMessage]) -> None:
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
    response_model=ChatCompletionResponse,
    summary="Obtém uma conclusão de chat.",
)
async def create_chat_completion(request: ChatCompletionRequest) -> JSONResponse:
    """
    Este endpoint fornece uma conclusão de chat com base na solicitação de chat fornecida.
    Retorna a resposta como StreamingResponse ou JSONResponse, dependendo da solicitação.
    """

    messages = request.data.messages
    if not messages:
        raise HTTPException(
            status_code=400, detail="A solicitação precisa conter mensagens."
        )

    # Processamento do serviço de completions
    chat_service = ChatService()
    chat_response = chat_service.get_chat_completion_v2(messages)

    # Se não for streaming, monta a resposta formatada como JSON
    return JSONResponse(
        content={
            "data": {
                "details": {
                    "idCompletion": chat_response.data.details.id_completion,
                    "objectType": "completion",
                    "createdDateTime": chat_response.data.details.created_date_time,
                    "usage": {
                        "completionTokenCount": chat_response.data.details.usage.completion_token_count,
                        "promptTokenCount": chat_response.data.details.usage.prompt_token_count,
                        "totalTokenCount": chat_response.data.details.usage.total_token_count,
                    },
                },
                "choices": [
                    {
                        "chatCompletionChoiceCommon": {
                            "indexOption": choice.chat_completion_choice_common.index_option,  # Acessando index_option corretamente
                            "finishReason": choice.chat_completion_choice_common.finish_reason,  # Acessando finish_reason corretamente
                        },
                        "message": {
                            "roleName": choice.message.role_name,  # Acessando role_name de ApiChatMessage
                            "messageContent": choice.message.message_content,  # Acessando message_content de ApiChatMessage
                            "endTurnIndicator": choice.message.end_turn_indicator
                            or True,  # Acessando end_turn_indicator de ApiChatMessage
                        },
                    }
                    for choice in chat_response.data.choices  # Acessando corretamente as choices dentro de data
                ],
            },
        },
        status_code=status.HTTP_200_OK,
    )
