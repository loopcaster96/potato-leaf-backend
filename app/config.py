"""
Módulo de configuración centralizada de la aplicación.

Utiliza Pydantic Settings (BaseSettings) para realizar la carga, el parseo
y la validación de tipos de todas las variables de entorno necesarias para
operar el backend. El uso de un Singleton instanciado a nivel de módulo
(`settings`) garantiza que la configuración se lea una única vez durante
el ciclo de vida del proceso, evitando lecturas repetidas del sistema de
archivos o del entorno en cada request.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Esquema fuertemente tipado de las variables de entorno del sistema.

    Cada atributo representa una variable de entorno obligatoria u opcional.
    Pydantic se encarga de la validación de tipos y de lanzar errores
    descriptivos en tiempo de arranque si una variable crítica no está
    presente, evitando fallos silenciosos en producción.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Configuración general de la aplicación
    # ------------------------------------------------------------------
    APP_NAME: str = "PotatoLeaf-AI Backend"
    APP_ENV: str = "production"
    API_V1_PREFIX: str = "/api/v1"

    # ------------------------------------------------------------------
    # Seguridad y Autenticación (JWT)
    # ------------------------------------------------------------------
    SECRET_KEY: str = Field(..., description="Clave secreta para firmar JWT")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 horas

    # ------------------------------------------------------------------
    # Autenticación Federada (Google OAuth2)
    # ------------------------------------------------------------------
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None

    # ------------------------------------------------------------------
    # Base de Datos (PostgreSQL Asíncrono)
    # ------------------------------------------------------------------
    DATABASE_URL: str = Field(
        ...,
        description="Cadena de conexión asíncrona, ej: "
        "postgresql+asyncpg://user:pass@db:5432/dbname",
    )

    # ------------------------------------------------------------------
    # Almacenamiento de Objetos (S3 / Cloudflare R2)
    # ------------------------------------------------------------------
    S3_ENDPOINT_URL: Optional[str] = None
    S3_PUBLIC_URL: Optional[str] = None
    S3_BUCKET_NAME: str = "potato-leaf-images"
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_REGION: str = "auto"

    # ------------------------------------------------------------------
    # Modelo de Machine Learning local (CNN)
    # ------------------------------------------------------------------
    MODEL_PATH: str = "models/cnn_plantvillage.keras"
    METADATA_PATH: str = "models/metadata_metrics.json"
    MODEL_INPUT_SIZE: int = 224
    LAST_CONV_LAYER_NAME: str = "auto"
    CLASS_NAMES: str = "Tizon_Temprano,Tizon_Tardio,Sana"

    # ------------------------------------------------------------------
    # Proveedores de LLM externos (Multi-Proveedor)
    # ------------------------------------------------------------------
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    AZURE_INFERENCE_ENDPOINT: Optional[str] = None
    AZURE_INFERENCE_CREDENTIAL: Optional[str] = None
    AZURE_DEPLOYMENT_NAME: str = "gpt-4o-mini"

    DEFAULT_LLM_PROVIDER: str = "gemini"

    # ------------------------------------------------------------------
    # CORS (Next.js Web + Aplicación Móvil)
    # ------------------------------------------------------------------
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:19006"

    @property
    def class_names_list(self) -> list[str]:
        """Convierte la lista de clases separada por comas en una lista de Python."""
        return [c.strip() for c in self.CLASS_NAMES.split(",") if c.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        """Convierte los orígenes CORS separados por comas en una lista de Python."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    """
    Provee una instancia única (cacheada) de Settings.

    El decorador lru_cache asegura que el objeto Settings se construya una
    sola vez por proceso, comportándose efectivamente como un Singleton
    inyectable a través del sistema de dependencias de FastAPI.
    """
    return Settings()


settings = get_settings()
