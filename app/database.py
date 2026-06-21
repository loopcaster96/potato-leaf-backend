"""
Módulo de conexión a la base de datos PostgreSQL mediante SQLAlchemy 2.0
en su modalidad completamente asíncrona (AsyncEngine + AsyncSession).

Se utiliza el driver `asyncpg` para garantizar que las operaciones de I/O
contra la base de datos no bloqueen el event loop de FastAPI, permitiendo
que el servidor procese múltiples requests concurrentes (incluyendo los
streams de los LLMs) sin degradar el rendimiento del servicio.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    """
    Clase base declarativa de SQLAlchemy 2.0.

    Todos los modelos ORM del sistema (`User`, `UserSettings`,
    `QueryHistory`, `LLMInterpretation`) heredan de esta clase para
    integrarse con el registro de metadatos compartido, requerido por
    Alembic y por la creación automática de tablas.
    """

    pass


engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependencia de FastAPI que provee una sesión de base de datos asíncrona
    por cada request entrante.

    El patrón de generador asíncrono garantiza el cierre determinista de
    la sesión (`async with`) incluso en presencia de excepciones, evitando
    fugas de conexiones del pool.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_models() -> None:
    """
    Crea el esquema de base de datos (todas las tablas) en el motor
    configurado, a partir de los metadatos registrados en `Base`.

    Se invoca durante el evento de arranque (`lifespan`) de la aplicación
    para garantizar que el esquema exista antes de aceptar tráfico,
    facilitando despliegues reproducibles en entornos de desarrollo y CI.
    En entornos productivos críticos se recomienda sustituir esta llamada
    por una migración gestionada con Alembic.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
