import pytest
from pydantic import ValidationError
from fastapi import HTTPException
from app.models.chat_models import ChatMessage, ChatRole
from app.routers.chat_routers import validate_messages, validate_user_header


def test_validate_messages_with_valid_messages() -> None:
    """
    Testa a função validate_messages com mensagens válidas.
    A função não deve lançar exceção.
    """

    # Arrange
    messages = [
        ChatMessage(content="start", role=ChatRole.SYSTEM),
        ChatMessage(content="Olá, como vai?", role=ChatRole.USER),
    ]

    # Act + Assert
    validate_messages(messages)  # Não deve lançar exceção


def test_validate_messages_with_empty_list() -> None:
    """
    Testa a função validate_messages com uma lista de mensagens vazia.
    A função deve lançar uma exceção HTTPException com status_code 400 e detail "A lista de mensagens não pode estar vazia."
    """

    # Arrange
    messages = []

    # Act
    with pytest.raises(HTTPException) as excp:
        validate_messages(messages)

    # Assert
    assert excp.value.status_code == 400
    assert excp.value.detail == "A lista de mensagens não pode estar vazia."


def test_validate_messages_with_empty_content() -> None:
    """
    Testa a função validate_messages com uma mensagem que não tem conteúdo.
    A função deve lançar uma exceção HTTPException com status_code 400 e detail "O atributo 'content' da mensagem no índice 0 não pode estar vazio."
    """

    # Arrange
    messages = [ChatMessage(content="", role=ChatRole.USER)]

    # Act
    with pytest.raises(HTTPException) as excp:
        validate_messages(messages)

    # Assert
    assert excp.value.status_code == 400
    assert (
        excp.value.detail
        == "O atributo 'content' da mensagem no índice 0 não pode estar vazio."
    )


def test_validate_messages_with_empty_role() -> None:
    """
    Testa a função validate_messages com uma mensagem que não tem role.
    A função deve lançar uma exceção HTTPException com status_code 400 e detail "O atributo 'role' da mensagem no índice 0 não pode estar vazio."
    """
    # Arrange
    messages = [dict(content="Olá, como vai?", role="")]

    # Act + Assert
    with pytest.raises(ValidationError):
        validate_messages([ChatMessage(**msg) for msg in messages])


def test_validate_messages_with_invalid_role() -> None:
    """
    Testa a função validate_messages com uma mensagem que tem um role inválido.
    A função deve lançar uma exceção HTTPException com status_code 400 e detail "O atributo 'role' da mensagem no índice 0 deve ser um dos seguintes: system, user, assistant"
    """
    # Arrange
    messages = [dict(content="Olá, como vai?", role="invalid_role")]

    # Act + Assert
    with pytest.raises(ValidationError):
        validate_messages([ChatMessage(**msg) for msg in messages])


def test_validate_user_header_with_valid_header() -> None:
    """
    Testa a função validate_user_header com um cabeçalho 'user-id' válido.
    A função não deve lançar exceção.
    """

    # Arrange
    user_id = "12345"

    # Act + Assert
    assert validate_user_header(user_id) == user_id  # Não deve lançar exceção


def test_validate_user_header_with_no_header() -> None:
    """
    Testa a função validate_user_header sem um cabeçalho 'user-id'.
    A função deve lançar uma exceção HTTPException com status_code 400 e detail "O header user-id é obrigatório."
    """

    # Arrange
    user_id = None

    # Act
    with pytest.raises(HTTPException) as excp:
        validate_user_header(user_id)

    # Assert
    assert excp.value.status_code == 400
    assert excp.value.detail == "O header user-id é obrigatório."
