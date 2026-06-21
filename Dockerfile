# =======================================================================
# Dockerfile de producción para el backend FastAPI de diagnóstico
# fitosanitario de papa (PlantVillage CNN + Multi-LLM).
#
# Se utiliza la imagen oficial python:3.10-slim como base, minimizando
# la superficie de ataque y el tamaño final de la imagen al evitar las
# herramientas de compilación y librerías de sistema innecesarias que
# trae la imagen completa de Debian.
# =======================================================================

FROM python:3.10-slim AS base

# -----------------------------------------------------------------------
# Variables de entorno de optimización del runtime de Python
# -----------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# -----------------------------------------------------------------------
# Dependencias de sistema mínimas requeridas por TensorFlow, Pillow y
# por la compilación de extensiones nativas de bcrypt/asyncpg.
# -----------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------
# Instalación de dependencias de Python.
# Se copia primero `requirements.txt` de forma aislada para maximizar
# el aprovechamiento de la cache de capas de Docker: mientras el código
# fuente cambia constantemente, las dependencias rara vez lo hacen.
# -----------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------
# Copia del código fuente de la aplicación y de los artefactos del
# modelo entrenado localmente.
# -----------------------------------------------------------------------
COPY app/ ./app/
COPY models/ ./models/

# -----------------------------------------------------------------------
# Creación de un usuario no privilegiado para ejecutar el proceso,
# siguiendo el principio de mínimo privilegio en entornos productivos.
# -----------------------------------------------------------------------
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# -----------------------------------------------------------------------
# Chequeo de salud a nivel de contenedor, consumido por Docker Compose
# y por orquestadores externos (Kubernetes, ECS) para determinar la
# disponibilidad real del proceso de Uvicorn.
# -----------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
