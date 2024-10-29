from enum import Enum


class SearchStrategy(str, Enum):
    """
    Enumeração que representa as estratégias de busca possíveis no serviço de chat.
    """

    HYBRID = "hybrid"
    SEMANTIC_HYBRID = "semantic_hybrid"
    SIMILARITY = "similarity"
