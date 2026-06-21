"""
Servicio Unificado de IA Agronómica — Capa de Abstracción con LiteLLM.

Este módulo reemplaza la dependencia de múltiples SDKs propietarios por
una única interfaz estandarizada provista por `litellm`, que actúa como
capa de traducción universal hacia cualquier proveedor de LLM externo
(Gemini, OpenAI, Anthropic/Claude, Groq, Azure OpenAI) sin modificar el
código del router que lo consume.

Arquitectura de la capa:

    DiagnoseRouter
        │
        ▼
    stream_agronomic_report()     ← punto de entrada público
        │
        ├─► _try_stream()         ← generador primario (modelo solicitado)
        │       └─ litellm.acompletion(stream=True)
        │
        └─► _try_stream()         ← generador de fallback (automático)
                └─ litellm.acompletion(stream=True)

Credenciales necesarias en `backend/.env`:
    GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    GROQ_API_KEY, AZURE_INFERENCE_ENDPOINT, AZURE_INFERENCE_CREDENTIAL.

LiteLLM resuelve el mapeo de credenciales leyendo las variables de entorno
estándar de cada proveedor (GEMINI_API_KEY → provider "gemini/...", etc.),
por lo que basta con exportarlas antes de iniciar el proceso.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import litellm
from litellm import ModelResponse
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
)

from app.config import settings
from app.schemas.schemas import StreamReportContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración global de LiteLLM
# ---------------------------------------------------------------------------
# Suprime la telemetría de LiteLLM para entornos de producción.
litellm.telemetry = False

# Tiempo de espera máximo por solicitud de streaming (segundos).
_REQUEST_TIMEOUT: int = 90

# Número de reintentos internos de LiteLLM antes de lanzar la excepción.
_MAX_RETRIES: int = 2

# Modelo de respaldo: se activa automáticamente si el modelo primario falla.
# Puede sobreescribirse vía DEFAULT_LLM_PROVIDER en .env.
_FALLBACK_MODEL: str = ""

# ---------------------------------------------------------------------------
# Prompt de sistema agronómico compartido
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT: str = (
    "Eres un ingeniero agrónomo experto en fitopatología de cultivos de papa "
    "(Solanum tuberosum), especializado en el diagnóstico de enfermedades "
    "foliares mediante visión computacional. Recibirás el veredicto "
    "probabilístico emitido por una red neuronal convolucional ya entrenada. "
    "Tu tarea es redactar un reporte interpretativo claro, técnico y "
    "accionable para un agricultor o técnico de campo, explicando la "
    "condición detectada, su origen biológico, el riesgo agronómico "
    "asociado y las recomendaciones de manejo fitosanitario pertinentes. "
    "Responde siempre en el idioma solicitado en el contexto."
)

# ---------------------------------------------------------------------------
# Mapeo de alias cortos → identificadores canónicos de LiteLLM
# ---------------------------------------------------------------------------
# Permite que el cliente pase alias simples ("gemini", "gpt4o", "claude")
# además de los identificadores completos ("gemini/gemini-2.0-flash").
_ALIAS_MAP: dict[str, str] = {
    # Google Gemini
    "gemini": "gemini/gemini-2.0-flash",
    "gemini-flash": "gemini/gemini-2.0-flash",
    "gemini-pro": "gemini/gemini-1.5-pro",
    # OpenAI
    "openai": "openai/gpt-4o-mini",
    "gpt4o": "openai/gpt-4o",
    "gpt4o-mini": "openai/gpt-4o-mini",
    # Anthropic / Claude
    "claude": "anthropic/claude-sonnet-4-6",
    "claude-sonnet": "anthropic/claude-sonnet-4-6",
    "claude-opus": "anthropic/claude-opus-4-6",
    # Groq
    "groq": "groq/llama-3.3-70b-versatile",
    "llama3": "groq/llama-3.3-70b-versatile",
    # Azure OpenAI  (usa AZURE_DEPLOYMENT_NAME como fallback de modelo)
    "azure": f"azure/{settings.AZURE_DEPLOYMENT_NAME}",
}


def _resolve_fallback_model() -> str:
    """
    Determina el modelo de respaldo a partir de DEFAULT_LLM_PROVIDER.

    Prioridad: si DEFAULT_LLM_PROVIDER ya contiene una barra ("/"), se
    interpreta como identificador completo de LiteLLM y se usa tal cual.
    En caso contrario, se resuelve a través del mapa de alias.
    """
    provider = settings.DEFAULT_LLM_PROVIDER.strip().lower()
    if "/" in provider:
        return provider
    return _ALIAS_MAP.get(provider, "gemini/gemini-2.0-flash")


# Inicializar correctamente el fallback (la función se llama antes de
# que el módulo termine de cargarse, así que se asigna aquí).
_FALLBACK_MODEL = _resolve_fallback_model()


def _resolve_model_name(model_name: str) -> str:
    """
    Traduce el identificador recibido desde el router al nombre canónico
    que LiteLLM espera para enrutar hacia el proveedor correcto.

    Acepta tanto alias cortos ("gemini", "claude") como identificadores
    completos con prefijo de proveedor ("groq/llama3-8b",
    "azure/gpt-4o-mini"). Los identificadores desconocidos se pasan
    directamente a LiteLLM para que intente la resolución por su cuenta.
    """
    normalized = model_name.strip().lower()
    if "/" in normalized:
        return normalized  # ya es un identificador canónico
    return _ALIAS_MAP.get(normalized, normalized)


def _build_messages(prompt: str, context: StreamReportContext) -> list[dict[str, str]]:
    """
    Construye el array de mensajes en formato OpenAI Chat Completions,
    compatible con LiteLLM para todos los proveedores soportados.

    El prompt base de interpretación se enriquece con el contexto
    agronómico estructurado del diagnóstico (veredicto CNN, confianza,
    geolocalización e idioma de respuesta requerido).
    """
    enriched_user_prompt = (
        f"{prompt}\n\n"
        f"--- Contexto del diagnóstico ---\n"
        f"Cultivo: {context.crop}\n"
        f"Veredicto de la red neuronal convolucional: {context.diagnostic_result}\n"
        f"Confianza del modelo: {context.confidence_score * 100:.2f}%\n"
        f"Ubicación geográfica de la captura: "
        f"lat={context.location_lat}, lon={context.location_lon}\n"
        f"Idioma de respuesta requerido: {context.language}\n"
        f"--- Fin del contexto ---\n"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": enriched_user_prompt},
    ]


def _litellm_extra_kwargs(model_name: str) -> dict[str, Any]:
    """
    Inyecta parámetros adicionales específicos del proveedor que LiteLLM
    no puede inferir automáticamente desde las variables de entorno.

    Actualmente necesario para Azure AI Inference, cuyo endpoint no sigue
    el esquema estándar de Azure OpenAI Service.
    """
    kwargs: dict[str, Any] = {}
    if model_name.startswith("azure/"):
        if settings.AZURE_INFERENCE_ENDPOINT:
            kwargs["api_base"] = settings.AZURE_INFERENCE_ENDPOINT
        if settings.AZURE_INFERENCE_CREDENTIAL:
            kwargs["api_key"] = settings.AZURE_INFERENCE_CREDENTIAL
    return kwargs


# ---------------------------------------------------------------------------
# Generador interno de streaming
# ---------------------------------------------------------------------------

async def _try_stream(
    model_name: str,
    messages: list[dict[str, str]],
) -> AsyncGenerator[str, None]:
    """
    Generador asíncrono interno que invoca `litellm.acompletion` en modo
    streaming y reemite cada fragmento de texto (token o chunk) recibido
    del proveedor.

    El parámetro `num_retries` de LiteLLM gestiona reintentos a nivel de
    red (errores 429, 503, timeouts transitorios). Los errores que
    persisten tras los reintentos se propagan hacia `stream_agronomic_report`
    para activar la lógica de conmutación de fallback.

    Args:
        model_name: Identificador canónico en formato "proveedor/modelo".
        messages:   Array de mensajes en formato Chat Completions.

    Yields:
        Fragmentos de texto (str) del reporte agronómico generado.

    Raises:
        APIConnectionError, RateLimitError, ServiceUnavailableError,
        AuthenticationError, APIError: en caso de fallo persistente del
        proveedor tras los reintentos configurados.
    """
    extra = _litellm_extra_kwargs(model_name)

    response = await litellm.acompletion(
        model=model_name,
        messages=messages,
        temperature=0.4,
        stream=True,
        timeout=_REQUEST_TIMEOUT,
        num_retries=_MAX_RETRIES,
        **extra,
    )

    async for chunk in response:
        # LiteLLM normaliza la estructura de chunks al formato OpenAI,
        # independientemente del proveedor subyacente.
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------

async def stream_agronomic_report(
    model_name: str,
    context: dict | StreamReportContext,
) -> AsyncGenerator[str, None]:
    """
    Genera de forma asíncrona el reporte agronómico interpretativo,
    transmitiendo los tokens en tiempo real hacia el router SSE.

    Implementa una política de fallback de dos niveles:

    1. **Modelo primario**: se intenta primero con el `model_name` recibido.
       LiteLLM gestiona internamente hasta `_MAX_RETRIES` reintentos ante
       fallos transitorios de red o límites de tasa.

    2. **Modelo de respaldo**: si el modelo primario falla de forma
       definitiva (autenticación inválida, proveedor caído, cuota agotada),
       el sistema conmuta automáticamente hacia `_FALLBACK_MODEL`, definido
       por `DEFAULT_LLM_PROVIDER` en `.env`, sin interrumpir la conexión
       SSE con el cliente. Se emite un evento de aviso antes del contenido
       de respaldo para que el cliente pueda notificar al usuario.

    Args:
        model_name: Identificador del modelo deseado. Acepta alias cortos
                    ("gemini", "claude", "groq") o nombres canónicos de
                    LiteLLM ("gemini/gemini-1.5-pro", "groq/llama3-8b").
        context:    Contexto del diagnóstico agronómico. Puede ser un dict
                    (deserializado desde JSON) o una instancia de
                    `StreamReportContext`.

    Yields:
        Fragmentos de texto (str) del reporte interpretativo. El generador
        también puede emitir líneas de metadatos con prefijo especial
        "__event:fallback_activated" para señalizar la conmutación al
        router SSE, que las convierte en eventos SSE dedicados.

    Notes:
        Esta función es el único punto de acoplamiento entre la capa de
        enrutamiento HTTP (`diagnose.py`) y la infraestructura de LLMs.
        El router no necesita conocer qué proveedor se está usando; sólo
        itera este generador y reemite cada fragmento al cliente.
    """
    # Normalizar el contexto: acepta dict o instancia Pydantic.
    if isinstance(context, dict):
        report_context = StreamReportContext(**context)
    else:
        report_context = context

    # Resolver alias y construir mensajes una sola vez para ambos intentos.
    resolved_model = _resolve_model_name(model_name)
    messages = _build_messages(
        prompt=(
            "Redacta un reporte agronómico interpretativo para el agricultor, "
            "basado estrictamente en el veredicto y el contexto suministrados "
            "a continuación. Estructura tu respuesta en tres secciones breves: "
            "(1) Diagnóstico y explicación biológica, (2) Riesgo y progresión "
            "esperada si no se interviene, (3) Recomendaciones de manejo "
            "fitosanitario inmediatas."
        ),
        context=report_context,
    )

    # -----------------------------------------------------------------------
    # Intento 1: Modelo primario solicitado
    # -----------------------------------------------------------------------
    try:
        logger.info(
            "LLMService | Iniciando stream con modelo primario: %s → %s",
            model_name,
            resolved_model,
        )
        async for token in _try_stream(resolved_model, messages):
            yield token
        logger.info(
            "LLMService | Stream completado exitosamente con: %s", resolved_model
        )
        return  # éxito: salir sin activar fallback

    except (
        APIConnectionError,
        RateLimitError,
        ServiceUnavailableError,
        AuthenticationError,
        APIError,
    ) as primary_error:
        logger.warning(
            "LLMService | Modelo primario '%s' falló: %s. "
            "Activando fallback hacia '%s'.",
            resolved_model,
            type(primary_error).__name__,
            _FALLBACK_MODEL,
        )

    except Exception as unexpected_error:
        # Captura errores no clasificados de LiteLLM o del proveedor.
        logger.exception(
            "LLMService | Error inesperado con modelo primario '%s': %s",
            resolved_model,
            unexpected_error,
        )

    # -----------------------------------------------------------------------
    # Intento 2: Fallback automático
    # -----------------------------------------------------------------------
    # Emitir señal de metadatos para que el router SSE la convierta en un
    # evento de aviso al cliente antes del contenido de respaldo.
    if resolved_model == _FALLBACK_MODEL:
        # Si el primario ya era el fallback, no hay segunda oportunidad.
        logger.error(
            "LLMService | El modelo primario y el fallback son el mismo (%s). "
            "No es posible la conmutación automática.",
            _FALLBACK_MODEL,
        )
        yield "__event:error|El proveedor seleccionado no está disponible y el modelo de respaldo coincide con el primario."
        return

    yield f"__event:fallback_activated|{resolved_model}→{_FALLBACK_MODEL}"

    try:
        logger.info(
            "LLMService | Iniciando stream con modelo de fallback: %s",
            _FALLBACK_MODEL,
        )
        async for token in _try_stream(_FALLBACK_MODEL, messages):
            yield token
        logger.info(
            "LLMService | Stream de fallback completado con: %s", _FALLBACK_MODEL
        )

    except Exception as fallback_error:
        logger.exception(
            "LLMService | El modelo de fallback '%s' también falló: %s",
            _FALLBACK_MODEL,
            fallback_error,
        )
        yield (
            f"__event:error|Tanto el modelo primario ({resolved_model}) como el "
            f"de respaldo ({_FALLBACK_MODEL}) fallaron. Intente más tarde."
        )
