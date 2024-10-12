import os
import tiktoken
from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from ..models.embedding_models import EmbeddingResponse


class EmbeddingService:
    """
    Representa o serviço responsável por gerar representações vetoriais (embeddings) de uma determinada entrada.
    """

    SYSTEM_MESSAGE_TEMPLATE = """
        Como assistente de IA, você deve ajudar instituições participantes dos mercados de atuação da B3, 
        que podem auxiliar os investidores e clientes na realização de negócios, a encontrar informações relevantes. 
        Responda as perguntas com base em {context}. Se a informação estiver em formato tabular, retorne-a como uma tabela HTML. 
        Se não puder responder com base nas informações disponíveis, informe que não sabe a resposta.
    """

    def __init__(self) -> None:
        """
        Inicializa o serviço de embeddings.

        Args:
            None

        Returns:
            None
        """

        load_dotenv(find_dotenv())
        self._init_env_vars()
        self._validate_env_vars()
        self._init_openai_embeddings()

    def create_embeddings(self, input_text: str) -> EmbeddingResponse:
        """
        Obtém embeddings para a entrada fornecida usando o modelo Azure OpenAI.

        Args:
            input_text (str): O texto de entrada para o qual os embeddings serão gerados.

        Returns:
            EmbeddingResponse: A resposta com os embeddings e o total de tokens.
        """

        token_count = 0
        if self._openai_embeddings.tiktoken_enabled:
            encoding = tiktoken.encoding_for_model(self._openai_embeddings.model)
            token_count = len(encoding.encode(input_text))

        embeddings = self._openai_embeddings.embed_query(input_text)
        embedding_response = EmbeddingResponse(
            embeddings=embeddings, total_tokens=token_count
        )

        return embedding_response

    def _init_env_vars(self) -> None:
        """
        Inicializa as variáveis de ambiente necessárias.

        Args:
            None

        Returns:
            None
        """

        self._openai_api_key = os.getenv("AZURE_API_KEY")
        self._openai_api_version = os.getenv("OPENAI_API_VERSION")
        self._openai_embeddings_deployment_name = os.getenv(
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"
        )
        self._openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    def _validate_env_vars(self) -> None:
        """
        Valida as variáveis de ambiente necessárias.

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: Se uma ou mais variáveis de ambiente obrigatórias não estiverem definidas.
        """

        required_vars = [
            self._openai_embeddings_deployment_name,
            self._openai_api_version,
            self._openai_endpoint,
            self._openai_api_key,
        ]

        if not all(required_vars):
            raise ValueError(
                "One or more Azure OpenAI environment variables are not set."
            )

    def _init_openai_embeddings(self) -> None:
        """
        Inicializa o Azure OpenAI Embeddings.

        Args:
            None

        Returns:
            None
        """

        self._openai_embeddings = AzureOpenAIEmbeddings(
            azure_deployment=self._openai_embeddings_deployment_name,
            openai_api_version=self._openai_api_version,
            azure_endpoint=self._openai_endpoint,
            api_key=self._openai_api_key,
        )
