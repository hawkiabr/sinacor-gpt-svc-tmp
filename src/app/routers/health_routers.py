"""
Este módulo define roteadores FastAPI para health checks.
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from opentelemetry.propagate import extract
from opentelemetry.trace import (
    SpanKind,
    get_tracer,
    get_tracer_provider,
)

from ..models.health_models import HealthCheckResponse

# Inicializando tracer e logger para OpenTelemetry
tracer = get_tracer(__name__, tracer_provider=get_tracer_provider())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Define o nível de log

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(request_data)s"
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Definindo o roteador de saúde
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
async def get_health_check(request: Request) -> JSONResponse:
    """
    Este endpoint verifica a saúde do serviço.
    Retorna a resposta como JSONResponse.
    """

    with tracer.start_as_current_span(
        "get_health_check", context=extract(request.headers), kind=SpanKind.SERVER
    ):
        try:
            # Status e versão do serviço
            service_status = "Healthy"
            dependencies_status = "Healthy"
            app_version = "v1.0.0"

            # Dados do health check para log em JSON
            log_data = {
                "event": "Health Check",
                "service_status": service_status,
                "dependencies_status": dependencies_status,
                "app_version": app_version,
            }

            # Registra o log em formato JSON
            logger.info(json.dumps(log_data))  # Registra como JSON

            # Retorna a resposta do health check
            return JSONResponse(
                {
                    "service_status": service_status,
                    "dependencies_status": dependencies_status,
                    "app_version": app_version,
                },
                status_code=status.HTTP_200_OK,
            )
        except Exception as e:
            # Log dos detalhes
            logging.error(
                "Erro ao processar o health check: %s",
                str(e),
                exc_info=True,
                extra={"request_data": request.model_dump()},
            )
            # Lança um erro genérico para problemas internos
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erro ao processar o health check.",
            ) from e
