"""
Esquemas de datos (DTOs) implementados con Pydantic v2.

Estos modelos desacoplan la representación interna de la base de datos
(SQLAlchemy ORM) de los contratos públicos expuestos por la API REST,
siguiendo el principio de separación de responsabilidades. Se utiliza
`model_config = ConfigDict(from_attributes=True)` para permitir la
serialización directa desde instancias ORM hacia los esquemas de
respuesta.
"""

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ----------------------------------------------------------------------
# Enumeraciones compartidas (replicadas en capa de transporte)
# ----------------------------------------------------------------------
class AuthProviderSchema(str, Enum):
    LOCAL = "local"
    GOOGLE = "google"


class DiagnosticResultSchema(str, Enum):
    SANA = "Sana"
    TIZON_TEMPRANO = "Tizon_Temprano"
    TIZON_TARDIO = "Tizon_Tardio"


class DeviceSourceSchema(str, Enum):
    WEB = "web"
    MOBILE = "mobile"


# ----------------------------------------------------------------------
# Esquemas de Autenticación
# ----------------------------------------------------------------------
class UserRegisterRequest(BaseModel):
    """Payload de entrada para el registro tradicional de usuarios."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


class UserLoginRequest(BaseModel):
    """Payload de entrada para el inicio de sesión tradicional."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class GoogleAuthRequest(BaseModel):
    """Payload de entrada para la autenticación federada con Google."""

    id_token: str = Field(
        ..., description="ID Token emitido por el SDK de Google en el cliente"
    )


class TokenResponse(BaseModel):
    """Respuesta estándar emitida tras una autenticación exitosa."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    """Estructura decodificada del payload interno del JWT."""

    sub: str
    exp: int


# ----------------------------------------------------------------------
# Esquemas de Usuario y Configuración
# ----------------------------------------------------------------------
class UserSettingsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    preferred_llm: str
    language: str
    notifications_enabled: bool


class UserSettingsUpdateRequest(BaseModel):
    """Payload de actualización parcial de las preferencias del usuario."""

    preferred_llm: str | None = Field(default=None, max_length=50)
    language: str | None = Field(default=None, max_length=10)
    notifications_enabled: bool | None = None


class UserProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    full_name: str | None
    auth_provider: AuthProviderSchema
    is_active: bool
    created_at: datetime
    settings: UserSettingsResponse | None = None


class UserProfileUpdateRequest(BaseModel):
    """Payload de actualización de los datos básicos del perfil."""

    full_name: str | None = Field(default=None, max_length=255)


# ----------------------------------------------------------------------
# Esquemas de Diagnóstico (CNN + Grad-CAM)
# ----------------------------------------------------------------------
# Se elimina GradCamMatrix, ahora se envía un PNG Base64


class DiagnoseResponse(BaseModel):
    """Respuesta completa emitida por el endpoint de diagnóstico CNN + XAI."""

    query_id: uuid.UUID
    diagnostic_result: DiagnosticResultSchema
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    heatmap_jet: str = Field(..., description="Mapa de calor puro (Jet colormap) en Base64")
    heatmap_overlay: str = Field(..., description="Mapa de calor superpuesto a la imagen en Base64")
    image_url: str
    location_lat: float
    location_lon: float
    created_at: datetime


class QueryHistoryItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    image_url: str
    diagnostic_result: DiagnosticResultSchema
    confidence_score: float
    location_lat: float
    location_lon: float
    device_source: DeviceSourceSchema
    created_at: datetime


# ----------------------------------------------------------------------
# Esquemas de Interpretación LLM
# ----------------------------------------------------------------------
class LLMInterpretationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    llm_provider: str
    generated_text: str
    created_at: datetime


class QueryHistoryDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    image_url: str
    diagnostic_result: DiagnosticResultSchema
    confidence_score: float
    location_lat: float
    location_lon: float
    device_source: DeviceSourceSchema
    created_at: datetime
    interpretations: list[LLMInterpretationResponse] = []


class StreamReportContext(BaseModel):
    """
    Contexto agronómico estructurado que se inyecta como prompt enriquecido
    a cada proveedor de LLM, garantizando que la interpretación generada
    sea coherente con el veredicto numérico emitido por la CNN local.
    """

    diagnostic_result: str
    confidence_score: float
    crop: str = "papa (Solanum tuberosum)"
    location_lat: float
    location_lon: float
    language: str = "es"
