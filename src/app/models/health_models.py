from typing import Optional
from pydantic import BaseModel


# TODO: Refatorar para incluir dependências
class HealthCheckResponse(BaseModel):
    """
    Modelo para a resposta do health check.
    """

    service_status: Optional[str] = None
    dependencies_status: Optional[str] = None
    app_version: Optional[str] = None
