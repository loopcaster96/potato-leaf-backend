"""
Router central del dominio de negocio: diagnóstico fitosanitario.

Expone dos endpoints complementarios pero independientes en el tiempo:

1. `/diagnose`: ejecuta la inferencia local de la CNN y el cálculo de
   Grad-CAM, persiste el registro geoespacial en PostgreSQL y retorna
   de inmediato el veredicto estructurado junto con la matriz de
   explicabilidad, sin esperar a ningún proveedor de IA externo.

2. `/diagnose/{query_id}/stream-report`: invocado posteriormente (de
   forma opcional y en caliente) por el cliente, una vez que el usuario
   ha seleccionado un proveedor de LLM en la interfaz. Este endpoint
   delega en `llm_service.stream_agronomic_report` (capa LiteLLM) la
   generación del reporte y transmite el flujo de tokens al cliente
   mediante Server-Sent Events (SSE). Incluye soporte para eventos
   de fallback y error diferenciados, permitiendo al cliente reaccionar
   a la conmutación automática de proveedor sin interrumpir el stream.

Eventos SSE emitidos por `/stream-report`:
    data: <token>             → fragmento de texto del reporte
    event: fallback_activated → el modelo primario falló; se usó el respaldo
    data: <from>→<to>         → detalle del proveedor primario y el de respaldo
    event: error              → error irrecuperable (ambos modelos fallaron)
    data: <mensaje>           → descripción del error
    event: done               → stream completado (éxito o con fallback)
    data: [DONE]
"""

import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, status, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.database_models import (
    DeviceSourceEnum,
    DiagnosticResultEnum,
    LLMInterpretation,
    QueryHistory,
    User,
)
from app.schemas.schemas import (
    DiagnoseResponse,
    QueryHistoryDetailResponse,
    StreamReportContext,
)
from app.services.llm_service import stream_agronomic_report
from app.services.ml_service import MLInferenceService
from app.services.storage_service import ObjectStorageService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/diagnose", tags=["Diagnóstico"])

def _get_ml_service(request: Request) -> MLInferenceService:
    """Recupera la instancia Singleton del servicio de inferencia CNN."""
    return request.app.state.ml_service


def _get_storage_service(request: Request) -> ObjectStorageService:
    """Recupera la instancia Singleton del servicio de almacenamiento."""
    return request.app.state.storage_service


@router.post(
    "",
    response_model=DiagnoseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ejecuta el diagnóstico CNN + Grad-CAM sobre una imagen foliar",
)
async def diagnose_leaf_image(
    request: Request,
    image: UploadFile = File(..., description="Fotografía de la hoja de papa"),
    lat: float = Form(..., description="Latitud geográfica de la captura"),
    lon: float = Form(..., description="Longitud geográfica de la captura"),
    device_source: str = Form(
        default="web", description="Origen del dispositivo: 'web' o 'mobile'"
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DiagnoseResponse:
    """
    Pipeline completo de diagnóstico: validación de entrada, subida del
    recurso visual al almacenamiento de objetos, inferencia local de la
    CNN, cálculo de Grad-CAM sobre la última capa convolucional, y
    persistencia transaccional del registro histórico geoespacial.
    """
    if device_source not in {item.value for item in DeviceSourceEnum}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="device_source debe ser 'web' o 'mobile'.",
        )

    content_type = image.content_type or "image/jpeg"
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo recibido no corresponde a una imagen válida.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo de imagen recibido está vacío.",
        )

    ml_service = _get_ml_service(request)
    storage_service = _get_storage_service(request)

    try:
        image_tensor = ml_service.preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No fue posible procesar la imagen suministrada: {exc}",
        ) from exc

    predicted_label, confidence_score, probability_map = ml_service.predict(
        image_tensor
    )
    predicted_index = ml_service.class_names.index(predicted_label)

    grad_cam_matrix = ml_service.compute_grad_cam(image_tensor, predicted_index)

    image_url = await storage_service.upload_image(
        file_bytes=image_bytes, content_type=content_type, user_id=str(current_user.id)
    )

    new_query = QueryHistory(
        user_id=current_user.id,
        image_url=image_url,
        diagnostic_result=DiagnosticResultEnum(predicted_label),
        confidence_score=confidence_score,
        location_lat=lat,
        location_lon=lon,
        device_source=DeviceSourceEnum(device_source),
    )
    db.add(new_query)
    await db.commit()
    await db.refresh(new_query)

    return DiagnoseResponse(
        query_id=new_query.id,
        diagnostic_result=predicted_label,
        confidence_score=confidence_score,
        probabilities=probability_map,
        heatmap_image=grad_cam_matrix,
        image_url=image_url,
        location_lat=lat,
        location_lon=lon,
        created_at=new_query.created_at,
    )


@router.get(
    "/{query_id}/stream-report",
    summary="Transmite en streaming el reporte interpretativo del LLM seleccionado",
)
async def stream_diagnosis_report(
    query_id: uuid.UUID,
    model: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Recupera el diagnóstico previamente persistido y genera en tiempo real
    el reporte interpretativo agronómico mediante la capa unificada de
    LiteLLM (`llm_service.stream_agronomic_report`).

    El parámetro `model` acepta tanto alias cortos ("gemini", "claude",
    "groq") como identificadores canónicos de LiteLLM
    ("gemini/gemini-1.5-pro", "azure/gpt-4o-mini"). Si el modelo
    solicitado falla, el sistema conmuta automáticamente al proveedor
    configurado en DEFAULT_LLM_PROVIDER, emitiendo un evento SSE
    `fallback_activated` para que el cliente pueda notificarlo al usuario.

    Eventos SSE emitidos:
        data: <token>              → fragmento de texto del reporte
        event: fallback_activated  → conmutación automática de proveedor
        data: <primario>→<respaldo>
        event: error               → error irrecuperable
        data: <descripción>
        event: done                → stream completado
        data: [DONE]

    El texto completo acumulado se persiste en `llm_interpretations` al
    finalizar el stream, sin bloquear la emisión de tokens al cliente.
    """
    result = await db.execute(
        select(QueryHistory).where(
            QueryHistory.id == query_id,
            QueryHistory.user_id == current_user.id,
        )
    )
    query_record = result.scalar_one_or_none()

    if query_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se encontró el registro de diagnóstico solicitado.",
        )

    context = StreamReportContext(
        diagnostic_result=query_record.diagnostic_result.value,
        confidence_score=query_record.confidence_score,
        location_lat=query_record.location_lat,
        location_lon=query_record.location_lon,
        language="es",
    )

    # Captura el modelo efectivo usado (puede ser el fallback) para la
    # persistencia posterior en llm_interpretations.
    effective_model_ref: list[str] = [model]

    async def _event_stream():
        """
        Generador SSE interno que itera el flujo de tokens del servicio
        unificado y traduce los metadatos de control (__event:*) en
        eventos SSE diferenciados, manteniendo el protocolo limpio
        frente al cliente (web o móvil).
        """
        accumulated_text: list[str] = []
        stream_error: bool = False

        try:
            async for chunk in stream_agronomic_report(
                model_name=model, context=context
            ):
                # ── Metadatos de control emitidos por llm_service ──────────
                if chunk.startswith("__event:fallback_activated|"):
                    # Extraer detalles "primario→respaldo" para el evento SSE.
                    detail = chunk.split("|", 1)[1]
                    # Actualizar referencia del modelo efectivo para persistencia.
                    parts = detail.split("→")
                    if len(parts) == 2:
                        effective_model_ref[0] = parts[1]
                    logger.warning(
                        "SSE | Fallback activado para query_id=%s: %s",
                        query_id,
                        detail,
                    )
                    yield f"event: fallback_activated\ndata: {detail}\n\n"
                    continue

                if chunk.startswith("__event:error|"):
                    error_msg = chunk.split("|", 1)[1]
                    logger.error(
                        "SSE | Error irrecuperable para query_id=%s: %s",
                        query_id,
                        error_msg,
                    )
                    yield f"event: error\ndata: {error_msg}\n\n"
                    stream_error = True
                    return

                # ── Token de contenido normal ───────────────────────────────
                accumulated_text.append(chunk)
                # Normalizar saltos de línea para el formato SSE.
                normalized = chunk.replace("\n", "\\n")
                yield f"data: {normalized}\n\n"

        except Exception as exc:
            logger.exception(
                "SSE | Excepción no controlada durante el stream para query_id=%s",
                query_id,
            )
            yield f"event: error\ndata: Error interno del servidor: {exc}\n\n"
            stream_error = True

        finally:
            yield "event: done\ndata: [DONE]\n\n"

        # ── Persistencia asíncrona del reporte completo ─────────────────────
        # Se ejecuta DESPUÉS de que el generador ha terminado de emitir,
        # por lo que no bloquea ningún fragmento enviado al cliente.
        if not stream_error:
            full_text = "".join(accumulated_text)
            if full_text.strip():
                try:
                    async with db.begin():
                        interpretation = LLMInterpretation(
                            query_id=query_record.id,
                            llm_provider=effective_model_ref[0],
                            generated_text=full_text,
                        )
                        db.add(interpretation)
                    logger.info(
                        "SSE | Reporte persistido para query_id=%s (modelo: %s)",
                        query_id,
                        effective_model_ref[0],
                    )
                except Exception as db_exc:
                    logger.exception(
                        "SSE | Error al persistir el reporte para query_id=%s: %s",
                        query_id,
                        db_exc,
                    )

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/{query_id}",
    response_model=QueryHistoryDetailResponse,
    summary="Obtiene el detalle completo de un diagnóstico y sus reportes de LLM",
)
async def get_query_detail(
    query_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> QueryHistory:
    """
    Recupera un diagnóstico por su ID, asegurando que pertenezca al usuario 
    autenticado. Retorna el resultado de la CNN y todos los reportes de LLM 
    generados previamente para esta consulta.
    """
    result = await db.execute(
        select(QueryHistory).where(
            QueryHistory.id == query_id,
            QueryHistory.user_id == current_user.id,
        )
    )
    query_record = result.scalar_one_or_none()

    if query_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se encontró el registro de diagnóstico solicitado o no tiene permisos.",
        )
    
    return query_record
