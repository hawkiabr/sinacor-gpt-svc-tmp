from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    """
    Enum para representar o papel de uma mensagem de chat.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """
    Modelo para representar uma mensagem de chat.
    """

    content: Optional[str] = None
    role: ChatRole  # TODO: personalizar as mensagens de erro do Pydantic, usadas pelo FastAPI para validação de dados


class ChatChoice(BaseModel):
    """
    Modelo para representar uma escolha de chat.
    """

    finish_reason: Optional[str] = None
    index: int
    message: ChatMessage


class ChatUsage(BaseModel):
    """
    Modelo para representar o uso de tokens de chat.
    """

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """
    Modelo para representar uma resposta de chat.
    """

    choices: List[ChatChoice]
    created: int
    id: Optional[str] = None
    usage: ChatUsage


class ChatRequest(BaseModel):
    """
    Modelo para representar uma solicitação de chat.
    """

    messages: List[ChatMessage]
    stream: bool = False


class Message(BaseModel):
    roleName: str = Field(
        ..., max_length=9, description="O papel do autor da mensagem."
    )
    messageContent: str = Field(
        ..., max_length=8000, description="Conteúdo da mensagem."
    )
    endTurnIndicator: Optional[bool] = Field(
        None, description="Indica se a mensagem termina o turno de bate-papo."
    )


class ChatCompletionRequestCommon(BaseModel):
    streamIndicator: bool = Field(
        default=False,
        description="Se verdadeiro, o fluxo de tokens será enviado à medida que estiverem disponíveis.",
    )
    userId: UUID = Field(..., description="Identificador do usuário")


class DataChatCompletionRequest(BaseModel):
    params: ChatCompletionRequestCommon
    messages: List[Message]


class ChatCompletionRequest(BaseModel):
    data: DataChatCompletionRequest


class ChatCompletionChoiceCommon(BaseModel):
    indexOption: int
    finishReason: Optional[str] = Field(
        None, description="Razão do término da resposta."
    )


class ChatCompletionChoice(BaseModel):
    chatCompletionChoiceCommon: ChatCompletionChoiceCommon
    message: Message


class Usage(BaseModel):
    completionTokenCount: int
    promptTokenCount: int
    totalTokenCount: int


class ChatCompletionResponseCommon(BaseModel):
    idCompletion: str
    objectType: str
    createdDateTime: int
    usage: Usage


class ChatCompletionResponse(BaseModel):
    data: dict
