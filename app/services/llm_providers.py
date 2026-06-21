"""
Implementaciones concretas de los proveedores de Large Language Models
y la fábrica (`LLMFactory`) que las orquesta de forma dinámica.

Cada clase concreta (`GeminiProvider`, `OpenAIProvider`, `ClaudeProvider`,
`GroqProvider`, `AzureProvider`) encapsula la inicialización perezosa de
su cliente asíncrono nativo respectivo y traduce el protocolo particular
de streaming de cada SDK comercial hacia la interfaz unificada definida
en `BaseLLMService`. De esta forma, el resto del sistema permanece
completamente agnóstico a las diferencias sintácticas entre SDKs.
"""

import logging
from collections.abc import AsyncGenerator

from anthropic import AsyncAnthropic
from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from fastapi import HTTPException, status
from google import genai
from google.genai import types as genai_types
from groq import AsyncGroq
from openai import AsyncOpenAI

from app.config import settings
from app.schemas.schemas import StreamReportContext
from app.services.llm_base import BaseLLMService

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_BASE = (
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


class GeminiProvider(BaseLLMService):
    """
    Proveedor concreto para Google Gemini, mediante el SDK unificado
    `google-genai`. Utiliza el método de streaming nativo
    `generate_content_stream` sobre el cliente asíncrono (`aio`).
    """

    provider_name = "gemini"

    def __init__(self, api_key: str | None):
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GEMINI_API_KEY no está configurada en el servidor.",
            )
        self._client = genai.Client(api_key=api_key)

    async def generate_stream_response(
        self, prompt: str, context: StreamReportContext
    ) -> AsyncGenerator[str, None]:
        full_prompt = self.build_prompt(prompt, context)
        stream = await self._client.aio.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT_BASE,
                temperature=0.4,
            ),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text


class OpenAIProvider(BaseLLMService):
    """
    Proveedor concreto para OpenAI nativo, mediante el SDK oficial
    `openai`, utilizando el cliente `AsyncOpenAI` y el endpoint de
    Chat Completions en modalidad streaming (`stream=True`).
    """

    provider_name = "openai"

    def __init__(self, api_key: str | None):
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OPENAI_API_KEY no está configurada en el servidor.",
            )
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate_stream_response(
        self, prompt: str, context: StreamReportContext
    ) -> AsyncGenerator[str, None]:
        full_prompt = self.build_prompt(prompt, context)
        stream = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_BASE},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.4,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class ClaudeProvider(BaseLLMService):
    """
    Proveedor concreto para Anthropic Claude, mediante el SDK oficial
    `anthropic`, utilizando el cliente `AsyncAnthropic` y su gestor de
    contexto nativo de streaming (`client.messages.stream`).
    """

    provider_name = "claude"

    def __init__(self, api_key: str | None):
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ANTHROPIC_API_KEY no está configurada en el servidor.",
            )
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate_stream_response(
        self, prompt: str, context: StreamReportContext
    ) -> AsyncGenerator[str, None]:
        full_prompt = self.build_prompt(prompt, context)
        async with self._client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_SYSTEM_PROMPT_BASE,
            messages=[{"role": "user", "content": full_prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


class GroqProvider(BaseLLMService):
    """
    Proveedor concreto para Groq (modelos Llama / Qwen de alta velocidad
    de inferencia), mediante el SDK oficial `groq` y su cliente
    `AsyncGroq`, replicando la interfaz de Chat Completions en streaming.
    """

    provider_name = "groq"

    def __init__(self, api_key: str | None):
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GROQ_API_KEY no está configurada en el servidor.",
            )
        self._client = AsyncGroq(api_key=api_key)

    async def generate_stream_response(
        self, prompt: str, context: StreamReportContext
    ) -> AsyncGenerator[str, None]:
        full_prompt = self.build_prompt(prompt, context)
        stream = await self._client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_BASE},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.4,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AzureProvider(BaseLLMService):
    """
    Proveedor concreto para Azure OpenAI / Azure AI Inference, mediante
    el SDK `azure-ai-inference`, utilizando el cliente asíncrono
    `ChatCompletionsClient` autenticado con `AzureKeyCredential`.
    """

    provider_name = "azure"

    def __init__(self, endpoint: str | None, credential: str | None, deployment: str):
        if not endpoint or not credential:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "AZURE_INFERENCE_ENDPOINT y AZURE_INFERENCE_CREDENTIAL "
                    "deben estar configuradas en el servidor."
                ),
            )
        self._client = ChatCompletionsClient(
            endpoint=endpoint, credential=AzureKeyCredential(credential)
        )
        self._deployment = deployment

    async def generate_stream_response(
        self, prompt: str, context: StreamReportContext
    ) -> AsyncGenerator[str, None]:
        full_prompt = self.build_prompt(prompt, context)
        stream = await self._client.complete(
            model=self._deployment,
            messages=[
                SystemMessage(content=_SYSTEM_PROMPT_BASE),
                UserMessage(content=full_prompt),
            ],
            temperature=0.4,
            stream=True,
        )
        async for update in stream:
            if update.choices and update.choices[0].delta.content:
                yield update.choices[0].delta.content


class LLMFactory:
    """
    Fábrica estática responsable de instanciar dinámicamente el proveedor
    de LLM solicitado por el cliente, materializando el Factory Pattern.

    Esta clase es el único punto de acoplamiento entre la capa de
    enrutamiento HTTP y las credenciales de configuración (`settings`),
    permitiendo que nuevos proveedores se incorporen al sistema
    extendiendo el diccionario `_REGISTRY` sin modificar el código de
    los routers que consumen la fábrica (Principio Abierto/Cerrado).
    """

    _REGISTRY = {
        "gemini",
        "openai",
        "claude",
        "groq",
        "azure",
    }

    @staticmethod
    def get_provider(provider_name: str) -> BaseLLMService:
        """
        Resuelve e instancia el proveedor concreto correspondiente al
        identificador textual recibido desde el query parameter
        `?provider=` del endpoint de streaming.

        Lanza una HTTPException 400 si el proveedor solicitado no existe
        en el registro, evitando fallos ambiguos más adelante en el
        pipeline de generación.
        """
        normalized_name = provider_name.strip().lower()

        if normalized_name not in LLMFactory._REGISTRY:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Proveedor de LLM '{provider_name}' no soportado. "
                    f"Proveedores disponibles: {sorted(LLMFactory._REGISTRY)}"
                ),
            )

        if normalized_name == "gemini":
            return GeminiProvider(api_key=settings.GEMINI_API_KEY)
        if normalized_name == "openai":
            return OpenAIProvider(api_key=settings.OPENAI_API_KEY)
        if normalized_name == "claude":
            return ClaudeProvider(api_key=settings.ANTHROPIC_API_KEY)
        if normalized_name == "groq":
            return GroqProvider(api_key=settings.GROQ_API_KEY)
        if normalized_name == "azure":
            return AzureProvider(
                endpoint=settings.AZURE_INFERENCE_ENDPOINT,
                credential=settings.AZURE_INFERENCE_CREDENTIAL,
                deployment=settings.AZURE_DEPLOYMENT_NAME,
            )

        # Rama defensiva: inalcanzable si _REGISTRY y este bloque están
        # sincronizados, pero se preserva para robustez ante refactors.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno de configuración de la fábrica de LLMs.",
        )
