from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from ..models.health_models import HealthCheckResponse

DEFAULT_MEDIA_TYPE = "application/json"

HealthRouter = APIRouter()


@HealthRouter.get(
    "/health",
    description="Este endpoint verifica a saúde do serviço.",
    operation_id="get_health_check",
    response_description="O status de saúde do serviço",
    response_model=HealthCheckResponse,
    summary="Verifica a saúde do serviço.",
    tags=["health"],
)
async def get_health_check() -> JSONResponse:
    """
    Este endpoint verifica a saúde do serviço.
    Retorna a resposta como JSONResponse.
    """

    # TODO: Adicione suas verificações de saúde aqui
    service_status = "Healthy"
    dependencies_status = "Healthy"
    app_version = "1.0.0"

    return JSONResponse(
        {
            "service_status": service_status,
            "dependencies_status": dependencies_status,
            "app_version": app_version,
        },
        media_type=DEFAULT_MEDIA_TYPE,
        status_code=status.HTTP_200_OK,
    )
