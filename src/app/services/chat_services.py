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

    def __init__(self) -> None:
        """
        Inicializa o serviço de chat.
        """

        load_dotenv(find_dotenv())
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
            raise ValueError("One or more environment variables are not set.")

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

        self._search_credential = AzureKeyCredential(self._azure_search_key)
        self._search_client = SearchClient(
            endpoint=self._azure_search_endpoint,
            index_name=self._azure_search_index_name,
            credential=self._search_credential,
        )

    def _init_openai_client(self) -> None:
        """
        Inicializa o Azure OpenAI.
        """

        self._openai_client = AzureChatOpenAI(
            azure_endpoint=self._azure_openai_endpoint,
            deployment_name=self._azure_openai_deployment_name,
            model=self._azure_openai_model,
            temperature=self._azure_openai_temperature,
            api_version=self._azure_openai_api_version,
            top_p=self._azure_openai_top_p,  # Incluindo top_p
        )

    def _init_openai_embeddings(self) -> None:
        """
        Inicializa o Azure OpenAI Embeddings.
        """

        self._openai_embeddings = AzureOpenAIEmbeddings(
            azure_deployment=self._azure_openai_embeddings_deployment_name,
            openai_api_version=self._azure_openai_api_version,
            azure_endpoint=self._azure_openai_endpoint,
            api_key=self._azure_openai_api_key,
        )

    def get_chat_completion(self, messages: List[ChatMessage]) -> ChatResponse:
        """
        Obtém uma resposta de chat (conclusão) usando modelo LLM.

        Args:
            messages (List[ChatMessage]): A lista de mensagens do chat para gerar a conclusão.

        Returns:
            ChatResponse: A resposta do chat gerada pelo modelo LLM.
        """

        context = self._retrieve_search_context(messages)
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

    def _retrieve_search_context(self, messages: List[ChatMessage]) -> str:
        """
        Busca o conteúdo apropriado do armazenamento e inclui referências.
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
        messages: List[ChatMessage],
        system_message: SystemMessage = None,
    ) -> ChatPromptValue:
        """
        Cria o prompt com base nas mensagens (perguntas).

        Args:
            context (str): O contexto a ser injetado no prompt.
            messages (List[ChatMessage]): A lista de mensagens do chat.
            system_message (SystemMessage, optional): A mensagem de sistema. Padrão é None.

        Returns:
            ChatPromptValue: O prompt criado com as mensagens e o contexto.
        """

        if system_message is None:
            system_message = self._create_system_message(context)

        user_message = HumanMessagePromptTemplate.from_template(messages[-1].content)
        prompt_template = ChatPromptTemplate.from_messages(
            [system_message, user_message]
        )

        return prompt_template.invoke({"context": context})
