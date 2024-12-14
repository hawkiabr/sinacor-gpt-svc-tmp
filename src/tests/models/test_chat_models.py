import pytest
from pydantic import ValidationError
from app.models.chat_models import ApiChatMessage


def test_chat_message_with_empty_role() -> None:
    """
    Testa a criação de um ChatMessage que não tem role.
    A criação deve lançar uma exceção ValidationError.
    """
    # Arrange
    invalid_message = dict(content="Olá, como vai?", role="")

    # Act and Assert
    with pytest.raises(ValidationError):
        ApiChatMessage(**invalid_message)


def test_chat_message_with_invalid_role() -> None:
    """
    Testa a criação de um ChatMessage com um role inválido.
    A criação deve lançar uma exceção ValidationError.
    """
    # Arrange
    invalid_message = dict(content="Olá, como vai?", role="invalid_role")

    # Act and Assert
    with pytest.raises(ValidationError):
        ApiChatMessage(**invalid_message)
