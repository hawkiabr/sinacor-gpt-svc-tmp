import os
import logging
import time

from typing import List, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..models.chat_models import (
    ChatChoice,
    ApiChatMessage,
    ApiChatUsage,
    ChatCompletionChoice,
    ChatCompletionChoiceCommon,
    ChatCompletionResponse,
    ChatCompletionResponseCommon,
    ChatMessage,
    ChatResponse,
    ChatRole,
    ChatUsage,
    DataChatCompletionResponse,
)


class ChatService:
    """
    Serviço responsável por gerenciar as interações de chat.
    Usa Chat Models para processar e responder às mensagens.
    """

    SYSTEM_MESSAGE_TEMPLATE = """
    Como assistente de IA, você deve ajudar no atendimento de chamados/tickets de suporte sobre o Sinacor, fornecendo informações relevantes para resolver as questões levantadas pelos usuários.

    - **Fontes de Informação**: Responda exclusivamente com base nas informações disponíveis no contexto fornecido em {context}. Se não houver informações suficientes para responder à pergunta, informe que não é possível responder com base nas informações disponíveis.

    - **Formato da Resposta**: Estruture a resposta da seguinte maneira:
        1. Responda diretamente à pergunta de forma clara e concisa, sempre em português (pt-br).
        2. Questões não relacionadas ao Sinacor não devem ser respondidas. Nesses casos, informe que não é possível responder com base nas informações disponíveis.
        3. Forneça detalhes relevantes em formato de lista, se necessário.
        4. Inclua a origem das informações na seção "Referências" do contexto fornecido. As referências devem ser listadas no seguinte formato: 'nome_do_arquivo#page=numero_da_página'. Por exemplo: 'ga prime - stvm.pdf#page=1'.
        5. Se não houver referências, informe que não é possível responder com base nas informações disponíveis.
        6. Certifique-se de que todas as respostas terminem com a seção de referências, conforme exemplo abaixo:

    - **Exemplo de Pergunta/Resposta**:
        ***Pergunta***: O que é o FixGear?
        
        ***Resposta***:
        O FixGear é um sistema do Sinacor responsável pela comunicação com o SMPFlash e pela recepção das mensagens, distribuindo-as para os sistemas que utilizarão os dados das mensagens.

            Referências:
            - ga prime - stvm.pdf#page=2
            - reinvestimento de tesouro direto.pdf#page=1

    Mantenha um tom impessoal, educado e profissional em todas as respostas, priorizando a clareza para auxiliar no atendimento ao chamado/ticket de suporte.
    """

    def __init__(
        self,
        message_history: Optional[BaseChatMessageHistory] = None,
    ) -> None:
        """
        Inicializa o serviço de chat.

        Args:
            message_history (Optional[BaseChatMessageHistory]): Histórico de mensagens do chat.
        """

        load_dotenv(find_dotenv())

        self._message_history = message_history
        self._init_env_vars()
        self._validate_env_vars()
        self._init_clients()

    def _init_env_vars(self) -> None:
        """
        Inicializa as variáveis de ambiente necessárias.
        """

        # Inicializando variáveis de ambiente para o cliente de busca
        self._azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        self._azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self._azure_search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        self._azure_search_top_results = int(os.getenv("AZURE_SEARCH_TOP_RESULTS", "3"))

        # Inicializando variáveis de ambiente para o cliente OpenAI
        self._azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self._azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self._azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self._azure_openai_model = os.getenv("AZURE_OPENAI_MODEL")
        self._azure_openai_temperature = float(
            os.getenv("AZURE_OPENAI_TEMPERATURE", "0.7")
        )
        self._azure_openai_embeddings_deployment_name = os.getenv(
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"
        )
        self._azure_openai_api_version = os.getenv("OPENAI_API_VERSION")
        self._azure_openai_api_type = os.getenv("OPENAI_API_TYPE", "azure")
        self._azure_openai_top_p = float(os.getenv("AZURE_OPENAI_TOP_P", "0.87"))

    def _validate_env_vars(self) -> None:
        """
        Valida as variáveis de ambiente necessárias.

        Raises:
            ValueError: Se uma ou mais variáveis de ambiente obrigatórias não estiverem definidas.
        """

        required_vars = [
            self._azure_search_index_name,
            self._azure_search_endpoint,
            self._azure_search_key,
            self._azure_openai_api_key,
            self._azure_openai_deployment_name,
            self._azure_openai_model,
            self._azure_openai_temperature,
            self._azure_openai_embeddings_deployment_name,
            self._azure_openai_api_version,
            self._azure_openai_api_type,
            self._azure_openai_top_p,
        ]

        if not all(required_vars):
            raise ValueError("Uma ou mais variáveis de ambiente não estão definidas.")

    def _init_clients(self) -> None:
        """
        Inicializa os clientes de busca e OpenAI.
        """

        self._init_search_client()
        self._init_openai_client()
        self._init_openai_embeddings()

    def _init_search_client(self) -> None:
        """
        Inicializa o cliente do Azure AI Search.
        """

        try:
            self._search_credential = AzureKeyCredential(self._azure_search_key)
            self._search_client = SearchClient(
                endpoint=self._azure_search_endpoint,
                index_name=self._azure_search_index_name,
                credential=self._search_credential,
            )

        except Exception as e:
            logging.error("Erro ao inicializar o cliente Azure AI Search: %s", e)
            raise

    def _init_openai_client(self) -> None:
        """
        Inicializa o cliente do Azure OpenAI.
        """

        try:
            self._openai_client = AzureChatOpenAI(
                azure_endpoint=self._azure_openai_endpoint,
                deployment_name=self._azure_openai_deployment_name,
                model=self._azure_openai_model,
                temperature=self._azure_openai_temperature,
                api_version=self._azure_openai_api_version,
                top_p=self._azure_openai_top_p,
            )

        except Exception as e:
            logging.error("Erro ao inicializar o cliente Azure OpenAI: %s", e)
            raise

    def _init_openai_embeddings(self) -> None:
        """
        Inicializa o cliente do Azure OpenAI Embeddings.
        """

        try:
            self._openai_embeddings = AzureOpenAIEmbeddings(
                azure_deployment=self._azure_openai_embeddings_deployment_name,
                openai_api_version=self._azure_openai_api_version,
                azure_endpoint=self._azure_openai_endpoint,
                api_key=self._azure_openai_api_key,
            )

        except Exception as e:
            logging.error(
                "Erro ao inicializar o cliente Azure OpenAI Embeddings: %s", e
            )
            raise

    def get_chat_completion(self, messages: List[ChatMessage]) -> ChatResponse:
        """
        Obtém uma resposta de chat (conclusão) usando o modelo LLM.

        Args:
            messages (List[ChatMessage]): A lista de mensagens do chat para gerar a conclusão.

        Returns:
            ChatResponse: A resposta do chat gerada pelo modelo LLM.
        """

        try:
            context = self._retrieve_search_context(messages)
            prompts = self._create_prompt(context)

            chain = prompts | self._openai_client

            history = []
            completion = chain.invoke(
                {"chat_history": history, "user_message": messages[-1].content}
            )

            if self._message_history:
                self._message_history.add_messages(
                    [
                        HumanMessage(content=messages[-1].content),
                        AIMessage(content=completion.content),
                    ]
                )

            chat_completion = ChatResponse(
                choices=[
                    ChatChoice(
                        finish_reason=completion.response_metadata["finish_reason"],
                        index=0,
                        message=ChatMessage(
                            content=completion.content,
                            role=ChatRole.ASSISTANT,
                        ),
                    ),
                ],
                created=int(time.time()),
                id=completion.id,
                usage=ChatUsage(
                    completion_tokens=completion.response_metadata["token_usage"][
                        "completion_tokens"
                    ],
                    prompt_tokens=completion.response_metadata["token_usage"][
                        "prompt_tokens"
                    ],
                    total_tokens=completion.response_metadata["token_usage"][
                        "total_tokens"
                    ],
                ),
            )

            return chat_completion

        except Exception as e:
            logging.error("Erro ao realizar a conclusão: %s", e)
            raise

    def get_chat_completion_v2(self, messages: List[ApiChatMessage]) -> ChatResponse:
        """
        Obtém uma resposta de chat (conclusão) usando o modelo LLM.

        Args:
            messages (List[ApiChatMessage]): A lista de mensagens no formato da API.

        Returns:
            ChatResponse: A resposta do chat gerada pelo modelo LLM.
        """

        try:
            # Converte as mensagens da API (ApiChatMessage) para o modelo interno ChatMessage
            chat_messages = [
                ChatMessage(role=msg.role_name, content=msg.message_content)
                for msg in messages
            ]

            # Agora, use o modelo ChatMessage para gerar a resposta
            context = self._retrieve_search_context(chat_messages)
            prompt = self._create_prompt(context)

            chain = prompt | self._openai_client

            history = []
            completion = chain.invoke(
                {"chat_history": history, "user_message": chat_messages[-1].content}
            )

            if self._message_history:
                self._message_history.add_messages(
                    [
                        HumanMessage(content=chat_messages[-1].content),
                        AIMessage(content=completion.content),
                    ]
                )

            chat_completion = ChatCompletionResponse(
                data=DataChatCompletionResponse(
                    details=ChatCompletionResponseCommon(
                        created_date_time=int(time.time()),
                        id_completion=completion.id,
                        usage=ApiChatUsage(
                            completion_token_count=completion.response_metadata[
                                "token_usage"
                            ]["completion_tokens"],
                            prompt_token_count=completion.response_metadata[
                                "token_usage"
                            ]["prompt_tokens"],
                            total_token_count=completion.response_metadata[
                                "token_usage"
                            ]["total_tokens"],
                        ),
                    ),
                    choices=[
                        ChatCompletionChoice(
                            chat_completion_choice_common=ChatCompletionChoiceCommon(
                                index_option=0,
                                finish_reason=completion.response_metadata[
                                    "finish_reason"
                                ],
                            ),
                            message=ApiChatMessage(
                                role_name=ChatRole.ASSISTANT,
                                message_content=completion.content,
                                end_turn_indicator=True,
                            ),
                        ),
                    ],
                ),
            )

            return chat_completion

        except Exception as e:
            logging.error("Erro ao realizar a conclusão: %s", e)
            raise

    def _retrieve_search_context(self, messages: List[ChatMessage]) -> str:
        """
        Busca o conteúdo apropriado do armazenamento e inclui referências.

        Args:
            messages (List[ChatMessage]): A lista de mensagens para buscar contexto relevante.

        Returns:
            str: O contexto de busca que será utilizado para gerar a resposta.
        """

        search_context = ""
        references = []

        for message in messages:
            results = self._search_client.search(
                search_text=message.content, top=self._azure_search_top_results
            )

            for doc in results:
                search_context += "\n" + doc["content"]
                if "sourcepage" in doc:
                    references.append(doc["sourcepage"])  # Captura o sourcepage

        if references:
            search_context += "\n\nReferências:\n" + "\n".join(references)

        return search_context.strip()

    def _invoke_openai_model(self, prompt: str) -> BaseMessage:
        """
        Invoca o modelo para gerar uma conclusão com base no prompt fornecido.

        Args:
            prompt (str): O prompt a ser enviado ao modelo OpenAI.

        Returns:
            BaseMessage: A mensagem gerada pelo modelo.
        """

        return self._openai_client.invoke(prompt)

    def _create_system_message(self, context: str = None) -> SystemMessage:
        """
        Cria uma mensagem de sistema com instruções para o assistente de IA.

        Args:
            context (str, optional): O contexto a ser injetado na mensagem. Padrão é "informações disponíveis".

        Returns:
            SystemMessage: A mensagem de sistema criada.
        """

        context = context or "informações disponíveis"
        return SystemMessage(
            content=self.SYSTEM_MESSAGE_TEMPLATE.format(context=context)
        )

    def _create_prompt(
        self,
        context: str,
        system_message: SystemMessage = None,
    ) -> ChatPromptTemplate:
        """
        Cria o prompt com base nas mensagens (perguntas).

        Args:
            context (str): O contexto a ser injetado no prompt.
            system_message (SystemMessage, optional): A mensagem de sistema. Padrão é None.

        Returns:
            ChatPromptTemplate: O template de prompt.
        """

        if system_message is None:
            system_message = self._create_system_message(context)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                system_message,
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{user_message}"),
            ]
        )

        return prompt_template
