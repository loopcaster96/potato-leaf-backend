"""
Contrato polimórfico base para los proveedores de Large Language Models.

`BaseLLMService` define la interfaz mínima e inmutable que toda
implementación concreta de proveedor debe satisfacer, permitiendo que la
capa de orquestación (`LLMFactory` y los routers) opere exclusivamente
contra esta abstracción sin conocer los detalles particulares de cada
SDK comercial (Gemini, OpenAI, Anthropic, Groq, Azure). Este desacoplamiento
es la base del Principio de Sustitución de Liskov aplicado a integraciones
de IA generativa heterogéneas.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from app.schemas.schemas import StreamReportContext


class BaseLLMService(ABC):
    """
    Interfaz abstracta que deben implementar todos los proveedores de LLM.

    El método `generate_stream_response` se define como un generador
    asíncrono (`AsyncGenerator[str, None]`), de modo que cada
    implementación concreta pueda emitir tokens de texto de forma
    incremental a medida que el proveedor externo los retorna, sin
    bloquear el event loop de FastAPI mientras se espera la respuesta
    completa.
    """

    provider_name: str = "base"

    @abstractmethod
    async def generate_stream_response(
        self, prompt: str, context: StreamReportContext
    ) -> AsyncGenerator[str, None]:
        """
        Genera de forma asíncrona un flujo de fragmentos de texto (tokens
        o chunks) correspondientes al reporte agronómico interpretativo.

        Args:
            prompt: Instrucción textual base que orienta la generación.
            context: Contexto estructurado del diagnóstico (veredicto de
                la CNN, score de confianza, ubicación geográfica e idioma)
                que enriquece al prompt para garantizar coherencia clínica
                entre el resultado numérico y la explicación en lenguaje
                natural.

        Yields:
            Fragmentos de texto (str) emitidos incrementalmente por el
            proveedor subyacente, listos para ser retransmitidos al
            cliente mediante Server-Sent Events (SSE).
        """
        raise NotImplementedError
        yield ""  # pragma: no cover - garantiza la firma de generador

    def build_prompt(self, base_prompt: str, context: StreamReportContext) -> str:
        """
        Construye el prompt final enriquecido con el contexto agronómico
        estructurado, compartido por todas las implementaciones concretas
        para garantizar consistencia narrativa entre proveedores.
        """
        return (
            f"{base_prompt}\n\n"
            f"--- Contexto del diagnóstico ---\n"
            f"Cultivo: {context.crop}\n"
            f"Veredicto de la red neuronal convolucional: {context.diagnostic_result}\n"
            f"Confianza del modelo: {context.confidence_score * 100:.2f}%\n"
            f"Ubicación geográfica de la captura: "
            f"lat={context.location_lat}, lon={context.location_lon}\n"
            f"Idioma de respuesta requerido: {context.language}\n"
            f"--- Fin del contexto ---\n"
        )
