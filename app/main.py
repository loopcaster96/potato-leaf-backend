"""
Punto de entrada principal de la aplicación FastAPI.

Define el ciclo de vida (`lifespan`) del servidor, responsable de la
carga única en memoria del modelo CNN y de la inicialización del
esquema de base de datos antes de aceptar tráfico entrante. Asimismo,
configura el middleware de CORS necesario para que tanto la aplicación
web (Next.js) como la aplicación móvil puedan consumir la API desde
orígenes distintos, y centraliza el registro de todos los routers del
dominio.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_models
from app.routers import auth, diagnose, users
from app.services.ml_service import build_ml_service
from app.services.storage_service import build_storage_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestor de contexto asíncrono que orquesta el arranque y el apagado
    ordenado del servidor.

    En el arranque: inicializa el esquema de base de datos, carga el
    modelo `cnn_plantvillage.keras` en memoria RAM exactamente una vez,
    y construye el servicio de almacenamiento de objetos, almacenando
    ambas instancias en `app.state` para su reutilización transversal
    por parte de los routers sin reconstrucción por request.
    """
    logger.info("Iniciando secuencia de arranque del backend...")

    await init_models()
    logger.info("Esquema de base de datos verificado/inicializado.")

    ml_service = build_ml_service()
    ml_service.load_model()
    app.state.ml_service = ml_service

    app.state.storage_service = build_storage_service()
    logger.info("Servicio de almacenamiento de objetos inicializado.")

    logger.info("Backend listo para aceptar tráfico.")
    yield

    logger.info("Apagando backend de forma ordenada...")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Backend centralizado para el diagnóstico temprano e interpretable "
        "de enfermedades foliares en el cultivo de papa, mediante CNN local "
        "con explicabilidad Grad-CAM y orquestación dinámica Multi-LLM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(diagnose.router, prefix=settings.API_V1_PREFIX)


@app.get("/health", tags=["Sistema"], summary="Chequeo de salud del servicio")
async def health_check() -> dict[str, str]:
    """
    Endpoint de verificación de disponibilidad, consumido por el
    `healthcheck` del contenedor Docker y por balanceadores de carga
    externos para determinar si la instancia está lista para recibir
    tráfico productivo.
    """
    return {"status": "ok", "service": settings.APP_NAME}
