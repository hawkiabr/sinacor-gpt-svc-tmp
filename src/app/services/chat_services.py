import os
import time
from typing import List
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from ..models.chat_models import (
    ChatChoice,
    ChatMessage,
    ChatResponse,
    ChatRole,
    ChatUsage,
)

from ..models.search_models import SearchStrategy


class ChatService:
    """
    Representa o serviço responsável por gerenciar as interações de chat.
    Usa Chat Models para processar e responder às mensagens.
    """

    SYSTEM_MESSAGE_TEMPLATE = """
        Como assistente de IA, você deve ajudar instituições participantes dos mercados de atuação da B3, 
        que podem auxiliar os investidores e clientes na realização de negócios, a encontrar informações relevantes. 
        Responda as perguntas com base em {context}. Se a informação estiver em formato tabular, retorne-a como uma tabela HTML. 
        Se não puder responder com base nas informações disponíveis, informe que não sabe a resposta.
        """

    def __init__(self) -> None:
        """
        Inicializa o serviço de chat.
        """

        load_dotenv(find_dotenv())
        self._init_search_client()
        self._init_openai_client()
        self._init_openai_embeddings()
        # self._init_vector_store()

    def get_chat_completion(self, messages: List[ChatMessage]) -> ChatResponse:
        """
        Obtém uma resposta de chat (conclusão) usando modelo LLM.
        """

        context = self._retrieve_search_context(messages)
        # context = self._retrieve_search_context_from_vector_store(messages)
        prompt = self._create_prompt(context, messages)
        completion = self._invoke_openai_model(prompt)

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

    def _init_search_client(self) -> None:
        """
        Inicializa o cliente do Azure AI Search.
        """

        self.search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_KEY")

        self.search_credential = AzureKeyCredential(self.search_key)
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.search_index_name,
            credential=self.search_credential,
        )

    def _init_openai_client(self) -> None:
        """
        Inicializa o Azure OpenAI.
        """

        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")

        self.openai_client = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
            model=os.getenv("OPENAI_MODEL"),
            temperature=os.getenv("OPENAI_TEMPERATURE") or 0.7,
        )

    def _init_openai_embeddings(self) -> None:
        """
        Inicializa o Azure OpenAI Embeddings.
        """

        self.openai_embeddings_deployment_name = os.getenv(
            "OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"
        )
        self.openai_api_version = os.getenv("OPENAI_API_VERSION")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        self.openai_embeddings = AzureOpenAIEmbeddings(
            azure_deployment=self.openai_embeddings_deployment_name,
            openai_api_version=self.openai_api_version,
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.openai_api_key,
        )

    def _init_vector_store(self) -> None:
        """
        Inicializa o cliente do Azure AI Search como um Vector Store.
        """

        self.vector_store_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        self.vector_store_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.vector_store_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

        self.vector_store = AzureSearch(
            azure_search_endpoint=self.vector_store_endpoint,
            azure_search_key=self.vector_store_key,
            index_name=self.vector_store_index_name,
            embedding_function=self.openai_embeddings.embed_query,
        )

    def _retrieve_search_context(self, messages: List[ChatMessage]) -> str:
        """
        Busca o conteúdo apropriado do armazenamento.
        """

        search_top_results = os.getenv("AZURE_SEARCH_TOP_RESULTS") or 3
        search_context = ""

        for message in messages:
            results = self.search_client.search(
                search_text=message.content, top=search_top_results
            )

            for doc in results:
                search_context += "\n" + doc["content"]

        return search_context

    def _retrieve_search_context_from_vector_store(
        self,
        messages: List[ChatMessage],
        strategy: SearchStrategy = SearchStrategy.SIMILARITY,
    ) -> str:
        """
        Busca o conteúdo apropriado do armazenamento.
        """

        search_top_results = os.getenv("AZURE_SEARCH_TOP_RESULTS") or 3
        search_context = ""

        for message in messages:

            if strategy == SearchStrategy.SIMILARITY:
                results = self.vector_store.vector_search(
                    query=message.content,
                    k=search_top_results,
                )
            elif strategy == SearchStrategy.HYBRID:
                results = self.vector_store.hybrid_search(
                    query=message.content,
                    k=search_top_results,
                )

            for doc in results:
                search_context += "\n" + doc.page_content

        return search_context

    def _invoke_openai_model(self, prompt: str) -> BaseMessage:
        """
        Invoca o modelo para gerar uma conclusão com base no prompt fornecido.
        """
        return self.openai_client.invoke(prompt)

    def _create_system_message(self, context: str = None) -> SystemMessage:
        """
        Cria uma mensagem de sistema com instruções para o assistente de IA.
        Permite a injeção de um contexto.
        """

        context = context or "informações disponíveis"
        return SystemMessage(
            content=self.SYSTEM_MESSAGE_TEMPLATE.format(context=context)
        )

    def _create_prompt(
        self,
        context: str,
        messages: List[ChatMessage],
        system_message: SystemMessage = None,
    ) -> ChatPromptValue:
        """
        Cria o prompt com base nas mensagens (perguntas).
        Permite a injeção de uma mensagem de sistema.
        """

        if system_message is None:
            system_message = self._create_system_message(context)

        user_message = HumanMessagePromptTemplate.from_template(messages[-1].content)

        prompt_template = ChatPromptTemplate.from_messages(
            [system_message, user_message]
        )

        return prompt_template.invoke({"context": context})
