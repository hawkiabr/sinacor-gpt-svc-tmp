"""
Este módulo define os roteadores FastAPI para interações e conclusões de chat.
"""

import logging
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


def _validate_messages(
    messages: List[ApiChatMessage], is_completion: bool = False
) -> None:
    """
    Valida uma lista de mensagens com base no tipo de endpoint.

    - Para o endpoint `/chat`, as mensagens devem conter os atributos 'content' e 'role'.
    - Para o endpoint `/chat/completion`, as mensagens devem conter 'roleName', 'messageContent' e 'endTurnIndicator'.

    Parâmetros:
    - messages: Lista de objetos `ApiChatMessage` a serem validados.
    - is_completion: Se `True`, valida para o endpoint `/chat/completion`, caso contrário, valida para o endpoint `/chat`.

    Lança uma HTTPException com todos os erros encontrados.
    """
    errors = []  # Lista para acumular os erros de validação

    if not messages:
        errors.append("A lista de mensagens não pode estar vazia.")

    for i, message in enumerate(messages):
        if is_completion:
            # Validação para o endpoint /chat/completion
            if not message.role_name:
                errors.append(
                    f"O atributo 'roleName' da mensagem no índice {i} não pode estar vazio para /chat/completion."
                )

            if not message.message_content:
                errors.append(
                    f"O atributo 'messageContent' da mensagem no índice {i} não pode estar vazio para /chat/completion."
                )

            if message.role_name not in ChatRole:
                errors.append(
                    f"O atributo 'roleName' da mensagem no índice {i} deve ser um dos seguintes: {', '.join([role.value for role in ChatRole])} para /chat/completion."
                )

            if message.end_turn_indicator is None:
                errors.append(
                    f"O atributo 'endTurnIndicator' da mensagem no índice {i} não pode ser vazio para /chat/completion."
                )
        else:
            # Validação para o endpoint /chat
            if not message.content:
                errors.append(
                    f"O atributo 'content' da mensagem no índice {i} não pode estar vazio para /chat."
                )

            if not message.role:
                errors.append(
                    f"O atributo 'role' da mensagem no índice {i} não pode estar vazio para /chat."
                )

            if message.role not in ChatRole:
                errors.append(
                    f"O atributo 'role' da mensagem no índice {i} deve ser um dos seguintes: {', '.join([role.value for role in ChatRole])} para /chat."
                )

    # Se houver algum erro, lança uma exceção com todos os erros encontrados
    if errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(errors),
        )


def _validate_user_header(user_id: str = Header(None)) -> str:
    """
    Valida a presença do cabeçalho 'user-id' na requisição HTTP.

    Parâmetros:
    - user_id: O valor do cabeçalho 'user-id', que é obrigatório.

    Retorna:
    - O valor do cabeçalho 'user-id' se presente.

    Lança uma HTTPException se o cabeçalho 'user-id' estiver ausente.
    """

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O cabeçalho 'user-id' é obrigatório.",
        )

    return user_id


# Definindo o roteador APIRouter
ChatRouter = APIRouter(
    dependencies=[Depends(_validate_user_header)],
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

    Parâmetros:
    - request: Objeto `ChatRequest` contendo a lista de mensagens e o atributo 'stream'.

    Retorna:
    - StreamingResponse ou JSONResponse, dependendo do atributo 'stream' na solicitação.

    Lança:
    - HTTPException em caso de erro de validação ou exceções internas.
    """
    try:
        # Valida as mensagens recebidas
        _validate_messages(request.messages)

        # Cria o serviço de chat para processar as mensagens
        chat_service = ChatService()
        chat_response = chat_service.get_chat_completion(request.messages)

        # Função geradora para a resposta como streaming
        async def response_generator():
            for choice in chat_response.choices:
                yield choice.message.content

        # Retorna a resposta dependendo do atributo 'stream'
        if request.stream:
            return StreamingResponse(
                response_generator(),
                status_code=status.HTTP_200_OK,
            )

        return JSONResponse(
            chat_response.dict(),
            status_code=status.HTTP_200_OK,
        )

    except HTTPException as e:
        # Re-raise na HTTPException capturada
        logging.error(
            "Erro ao processar a solicitação de chat: %s",
            str(e),
            exc_info=True,
            extra={"request_data": request.model_dump()},
        )

        raise e

    except Exception as e:
        # Log dos detalhes
        logging.error(
            "Erro ao processar a solicitação de chat: %s",
            str(e),
            exc_info=True,
            extra={"request_data": request.model_dump()},
        )
        # Lança um erro genérico para problemas internos
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao processar a solicitação de chat.",
        ) from e


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

    Parâmetros:
    - request: Objeto `ChatCompletionRequest` contendo os dados e as mensagens a serem processadas.

    Retorna:
    - JSONResponse com a conclusão gerada.

    Lança:
    - HTTPException em caso de erro de validação ou exceções internas.
    """

    try:
        # Valida as mensagens para /chat/completion
        _validate_messages(request.data.messages, is_completion=True)

        # Processa a solicitação de conclusão de chat
        chat_service = ChatService()
        chat_response = chat_service.get_chat_completion_v2(request.data.messages)

        # Monta a estrutura de resposta
        response_content = {
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
                            "indexOption": choice.chat_completion_choice_common.index_option,
                            "finishReason": choice.chat_completion_choice_common.finish_reason,
                        },
                        "message": {
                            "roleName": choice.message.role_name,
                            "messageContent": choice.message.message_content,
                            "endTurnIndicator": choice.message.end_turn_indicator
                            or True,
                        },
                    }
                    for choice in chat_response.data.choices
                ],
            }
        }

        return JSONResponse(content=response_content, status_code=status.HTTP_200_OK)

    except HTTPException as e:
        # Log dos detalhes e re-raise na HTTPException capturada
        logging.error(
            "Erro ao processar a solicitação de conclusão de chat: %s",
            str(e),
            exc_info=True,
            extra={"request_data": request.model_dump()},
        )

        raise e

    except Exception as e:
        # Log dos detalhes
        logging.error(
            "Erro ao processar a solicitação de conclusão de chat: %s",
            str(e),
            exc_info=True,
            extra={"request_data": request.model_dump()},
        )
        # Lança um erro genérico para problemas internos
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao processar a solicitação de conclusão de chat.",
        ) from e
