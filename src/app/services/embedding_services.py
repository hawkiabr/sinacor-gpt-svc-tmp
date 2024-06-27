import os
import tiktoken
from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from ..models.embedding_models import EmbeddingResponse


class EmbeddingService:
    """
    Representa o serviço responsável por gerar representações vetoriais (embeddings) de uma determinada entrada.
    """

    def __init__(self) -> None:
        """
        Inicializa o serviço de embeddings.
        """

        load_dotenv(find_dotenv())
        self._init_openai_embeddings()

    def create_embeddings(
        self,
        input_text: str,
    ) -> EmbeddingResponse:
        """
        Obtém embeddings para a entrada fornecida usando o modelo Azure OpenAI.
        """

        if self.openai_embeddings.tiktoken_enabled:
            encoding = tiktoken.encoding_for_model(self.openai_embeddings.model)
            token_count = len(encoding.encode(input_text))

        embeddings = self.openai_embeddings.embed_query(input_text)
        embedding_response = EmbeddingResponse(
            embeddings=embeddings, total_tokens=token_count
        )

        return embedding_response

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
