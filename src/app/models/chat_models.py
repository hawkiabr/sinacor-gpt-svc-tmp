from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


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
