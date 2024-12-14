from enum import Enum
from typing import List, Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    """
    Enum para representar o papel de uma mensagem de chat.
    Os papéis podem ser:

    - 'user': O usuário que envia a mensagem.
    - 'assistant': O assistente que responde à mensagem.
    - 'system': O sistema que define o contexto ou comportamento.
    - 'ai': IA (Inteligência Artificial).
    - 'tool': Ferramenta utilizada no contexto do chat.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    AI = "ai"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """
    Modelo para representar uma mensagem de chat.
    """

    content: Optional[str] = None
    role: ChatRole


class ApiChatMessage(BaseModel):
    """
    Modelo para representar uma mensagem de chat.
    Contém o conteúdo da mensagem e o papel (role) do autor da mensagem.

    Atributos:
    - role_name: O papel do autor desta mensagem (ex: 'user', 'assistant').
    - message_content: O conteúdo da mensagem (até 8.000 caracteres).
    - end_turn_indicator: Indicador opcional se a mensagem termina o turno do bate-papo.
    """

    role_name: ChatRole = Field(
        ..., description="O papel do autor desta mensagem.", alias="roleName"
    )
    message_content: str = Field(
        ...,
        description="O conteúdo da mensagem (até 8.000 caracteres).",
        max_length=8000,
        min_length=1,
        alias="messageContent",
    )
    end_turn_indicator: Optional[bool] = Field(
        True,
        description="Se a mensagem termina o turno de bate-papo.",
        alias="endTurnIndicator",
    )

    class Config:
        """
        Configurações de alias para todos os campos do modelo
        """

        populate_by_name = True


class ChatChoice(BaseModel):
    """
    Modelo para representar uma escolha de chat.
    Cada escolha contém a mensagem gerada e a razão do término.

    Atributos:
    - finish_reason: Razão pela qual a resposta foi concluída.
    - index: Índice da escolha na lista de opções geradas.
    - message: A mensagem gerada para essa escolha.
    """

    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None
    index: int
    message: ChatMessage


class ChatUsage(BaseModel):
    """
    Modelo para representar o uso de tokens de chat.
    Contém a contagem de tokens utilizados durante o processamento da conclusão de chat.

    Atributos:
    - completion_tokens: Número de tokens usados na conclusão.
    - prompt_tokens: Número de tokens usados no prompt (entrada).
    - total_tokens: Número total de tokens utilizados.
    """

    completion_tokens: int = Field(..., alias="completionTokens")
    prompt_tokens: int = Field(..., alias="promptTokens")
    total_tokens: int = Field(..., alias="totalTokens")

    class Config:
        """
        Configurações de alias para todos os campos do modelo
        """

        populate_by_name = True  # Permite popular os campos pelos nomes ou aliases


class ApiChatUsage(BaseModel):
    """
    Modelo para representar o uso de tokens de chat.
    Contém a contagem de tokens utilizados durante o processamento da conclusão de chat.

    Atributos:
    - completion_token_count: Número de tokens usados na conclusão.
    - prompt_token_count: Número de tokens usados no prompt (entrada).
    - total_token_count: Número total de tokens utilizados.
    """

    completion_token_count: int = Field(..., alias="completionTokenCount")
    prompt_token_count: int = Field(..., alias="promptTokenCount")
    total_token_count: int = Field(..., alias="totalTokenCount")

    class Config:
        """
        Configurações de alias para todos os campos do modelo
        """

        populate_by_name = True  # Permite popular os campos pelos nomes ou aliases


class ChatRequest(BaseModel):
    """
    Modelo para representar uma solicitação de chat.
    Contém as mensagens enviadas pelo usuário e a indicação de se a resposta será em streaming.

    Atributos:
    - messages: Lista de mensagens que serão processadas no chat.
    - stream: Indicador se a resposta será retornada de forma contínua (streaming).
    """

    messages: List[ChatMessage]
    stream: bool = Field(False)


class ChatResponse(BaseModel):
    """
    Modelo para representar uma resposta de chat.
    A resposta inclui as escolhas geradas, a data de criação, o ID da resposta e o uso de tokens.

    Atributos:
    - choices: Lista de escolhas geradas pelo sistema.
    - created: Timestamp de criação da resposta.
    - id: ID da resposta gerada.
    - usage: Detalhes sobre o uso de tokens.
    """

    choices: List[ChatChoice]
    created: int = Field(..., alias="createdDateTime")
    id: Optional[str] = None
    usage: ChatUsage

    class Config:
        """
        Configurações de alias para todos os campos do modelo
        """

        populate_by_name = True  # Permite popular os campos pelos nomes ou aliases


class ChatCompletionRequestCommon(BaseModel):
    """
    Modelo para representar os parâmetros comuns de uma solicitação de conclusão de chat.
    Inclui a indicação de streaming e o ID do usuário.

    Atributos:
    - stream_indicator: Se verdadeiro, o fluxo de tokens será enviado à medida que estiverem disponíveis.
    - user_id: Identificador do usuário que está realizando a solicitação.
    """

    stream_indicator: bool = Field(default=False, alias="streamIndicator")
    user_id: UUID = Field(..., alias="userId")

    class Config:
        """
        Configurações de alias para todos os campos do modelo
        """

        populate_by_name = True  # Permite popular os campos pelos nomes ou aliases


class DataChatCompletionRequest(BaseModel):
    """
    Modelo para representar os dados de uma solicitação de conclusão de chat.
    Inclui os parâmetros e as mensagens a serem processadas.

    Atributos:
    - params: Parâmetros adicionais da solicitação, como 'stream_indicator' e 'user_id'.
    - messages: Lista de mensagens a serem processadas.
    """

    params: ChatCompletionRequestCommon
    messages: List[ApiChatMessage]


class ChatCompletionRequest(BaseModel):
    """
    Modelo para representar uma solicitação de conclusão de chat.
    Contém os dados necessários para gerar a conclusão de chat.

    Atributos:
    - data: Dados necessários para a solicitação de conclusão de chat.
    """

    data: DataChatCompletionRequest


class ChatCompletionChoiceCommon(BaseModel):
    """
    Modelo para representar os detalhes comuns de uma escolha de chat.
    Contém o índice da escolha e a razão do término da resposta.

    Atributos:
    - index_option: Índice da escolha na lista de opções geradas.
    - finish_reason: Razão do término da resposta (ex: "stop", "length", "content_filter").
    """

    index_option: int = Field(..., alias="indexOption")
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = Field(
        None, alias="finishReason"
    )

    class Config:
        """
        Configurações de alias para todos os campos do modelo
        """

        populate_by_name = True  # Permite popular os campos pelos nomes ou aliases


class ChatCompletionChoice(BaseModel):
    """
    Modelo para representar uma escolha completa de chat, incluindo os detalhes comuns e a mensagem.

    Atributos:
    - chat_completion_choice_common: Detalhes comuns sobre a escolha.
    - message: A mensagem gerada para essa escolha.
    """

    chat_completion_choice_common: ChatCompletionChoiceCommon = Field(
        ..., alias="chatCompletionChoiceCommon"
    )
    message: ApiChatMessage

    class Config:
        """
        Configurações de alias para todos os campos do modelo
        """

        populate_by_name = True  # Permite popular os campos pelos nomes ou aliases


class ChatCompletionResponseCommon(BaseModel):
    """
    Modelo para representar os detalhes comuns de uma resposta de conclusão de chat.

    Atributos:
    - id_completion: Identificador único da conclusão.
    - object_type: Tipo do objeto gerado (sempre "completion").
    - created_date_time: Timestamp de criação da conclusão.
    - usage: Detalhes sobre o uso de tokens.
    """

    id_completion: str = Field(..., alias="idCompletion")
    object_type: str = Field("completion", alias="objectType")
    created_date_time: int = Field(..., alias="createdDateTime")
    usage: ApiChatUsage

    class Config:
        """
        Configurações de alias para todos os campos do modelo
        """

        populate_by_name = True  # Permite popular os campos pelos nomes ou aliases


class DataChatCompletionResponse(BaseModel):
    """
    Modelo para representar os dados de uma resposta de conclusão de chat.

    Atributos:
    - details: Detalhes da conclusão, incluindo ID, tipo de objeto, criação e uso de tokens.
    - choices: Lista de escolhas geradas pelo sistema.
    """

    details: ChatCompletionResponseCommon
    choices: List[ChatCompletionChoice]

    class Config:
        """
        Configurações de alias para todos os campos do modelo
        """

        populate_by_name = True  # Permite popular os campos pelos nomes ou aliases


class ChatCompletionResponse(BaseModel):
    """
    Modelo para representar a resposta final de uma solicitação de conclusão de chat.
    Contém os dados da resposta com as escolhas geradas.

    Atributos:
    - data: Dados detalhados da resposta, incluindo escolhas geradas e detalhes sobre o uso de tokens.
    """

    data: DataChatCompletionResponse
