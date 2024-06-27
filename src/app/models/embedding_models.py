from typing import List, Optional, Union
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    """
    Modelo para representar uma solicitação de embeddings.
    """

    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None


class EmbeddingResponse(BaseModel):
    """
    Modelo para representar uma resposta de embeddings.
    """

    embeddings: List[float]
    total_tokens: int
