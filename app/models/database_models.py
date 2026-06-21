"""
Modelos ORM declarativos de SQLAlchemy 2.0.

Se utiliza la sintaxis moderna basada en `Mapped` y `mapped_column`,
que provee inferencia de tipos estática completa y es el estándar
recomendado por SQLAlchemy 2.0 para nuevos desarrollos. Las relaciones
entre entidades están definidas mediante `relationship` con carga
perezosa (`lazy="selectin"`) optimizada para el contexto asíncrono,
evitando el problema clásico de N+1 queries en escenarios de listados.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class AuthProviderEnum(str, enum.Enum):
    """Enumeración de los proveedores de autenticación soportados."""

    LOCAL = "local"
    GOOGLE = "google"


class DiagnosticResultEnum(str, enum.Enum):
    """Enumeración de los veredictos posibles emitidos por la CNN."""

    SANA = "Sana"
    TIZON_TEMPRANO = "Tizon_Temprano"
    TIZON_TARDIO = "Tizon_Tardio"


class DeviceSourceEnum(str, enum.Enum):
    """Enumeración del dispositivo cliente que originó la consulta."""

    WEB = "web"
    MOBILE = "mobile"


class User(Base):
    """
    Entidad de usuario del sistema.

    Soporta dos flujos de autenticación de forma excluyente y controlada
    por `auth_provider`: autenticación local (correo/contraseña hasheada
    con bcrypt) y autenticación federada vía Google OAuth2, en cuyo caso
    `hashed_password` permanece nulo por diseño.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    auth_provider: Mapped[AuthProviderEnum] = mapped_column(
        Enum(AuthProviderEnum, name="auth_provider_enum", native_enum=True),
        default=AuthProviderEnum.LOCAL,
        nullable=False,
    )
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    settings: Mapped["UserSettings"] = relationship(
        "UserSettings",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    queries: Mapped[list["QueryHistory"]] = relationship(
        "QueryHistory",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class UserSettings(Base):
    """
    Configuración personalizada y preferencias de cada usuario.

    Mantiene el proveedor de LLM por defecto seleccionado, el idioma de
    la interfaz y el estado de las notificaciones push/email.
    """

    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    preferred_llm: Mapped[str] = mapped_column(
        String(50), default="gemini", nullable=False
    )
    language: Mapped[str] = mapped_column(String(10), default="es", nullable=False)
    notifications_enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )

    user: Mapped["User"] = relationship(
        "User", back_populates="settings", lazy="selectin"
    )


class QueryHistory(Base):
    """
    Registro histórico geoespacial de cada diagnóstico realizado.

    Persiste la URL del recurso visual en el bucket de almacenamiento de
    objetos, el veredicto probabilístico emitido por la CNN, las
    coordenadas geográficas exactas de captura, y la trazabilidad
    temporal y de origen del dispositivo cliente.
    """

    __tablename__ = "queries_history"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_url: Mapped[str] = mapped_column(Text, nullable=False)
    diagnostic_result: Mapped[DiagnosticResultEnum] = mapped_column(
        Enum(DiagnosticResultEnum, name="diagnostic_result_enum", native_enum=True),
        nullable=False,
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    location_lat: Mapped[float] = mapped_column(Float, nullable=False)
    location_lon: Mapped[float] = mapped_column(Float, nullable=False)
    device_source: Mapped[DeviceSourceEnum] = mapped_column(
        Enum(DeviceSourceEnum, name="device_source_enum", native_enum=True),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(
        "User", back_populates="queries", lazy="selectin"
    )
    interpretations: Mapped[list["LLMInterpretation"]] = relationship(
        "LLMInterpretation",
        back_populates="query",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class LLMInterpretation(Base):
    """
    Reporte interpretativo generado por un proveedor de LLM específico
    para una consulta de diagnóstico determinada.

    Una misma `QueryHistory` puede tener múltiples interpretaciones
    asociadas, ya que el usuario puede solicitar reportes de distintos
    proveedores (Gemini, OpenAI, Claude, Groq, Azure) sobre el mismo
    diagnóstico sin necesidad de re-ejecutar la inferencia de la CNN.
    """

    __tablename__ = "llm_interpretations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("queries_history.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    llm_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    generated_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    query: Mapped["QueryHistory"] = relationship(
        "QueryHistory", back_populates="interpretations", lazy="selectin"
    )
