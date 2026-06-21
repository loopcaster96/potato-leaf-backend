# 🥔 PotatoLeaf-AI Backend

Backend centralizado para el **diagnóstico temprano e interpretable de enfermedades foliares en el cultivo de papa** (*Solanum tuberosum*), desarrollado como proyecto de tesis en Ingeniería de Sistemas. El sistema combina una Red Neuronal Convolucional (CNN) entrenada localmente sobre el dataset **PlantVillage**, explicabilidad mediante **Grad-CAM**, y una capa de orquestación **Multi-LLM unificada con LiteLLM** que genera reportes agronómicos interpretativos en streaming desde cualquiera de los cinco proveedores comerciales de IA soportados, con conmutación automática de fallback.

El backend alimenta concurrentemente una aplicación web (**Next.js**) y una aplicación móvil bajo un único contrato de API REST.

---

## 📋 Tabla de Contenidos

- [Arquitectura](#-arquitectura)
- [Pilares Funcionales](#-pilares-funcionales)
- [Stack Tecnológico](#-stack-tecnológico)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos Previos](#-requisitos-previos)
- [Instalación y Despliegue](#-instalación-y-despliegue)
- [Variables de Entorno](#-variables-de-entorno)
- [Modelo de Base de Datos](#-modelo-de-base-de-datos)
- [Documentación de la API](#-documentación-de-la-api)
- [Arquitectura Multi-LLM (LiteLLM + Fallback)](#-arquitectura-multi-llm-litellm--fallback)
- [Explicabilidad: Grad-CAM](#-explicabilidad-grad-cam)
- [Generación Automatizada del Proyecto](#-generación-automatizada-del-proyecto)
- [Troubleshooting](#-troubleshooting)

---

## 🏗️ Arquitectura

El sistema sigue una arquitectura en capas (*layered architecture*) desacoplada mediante inyección de dependencias nativa de FastAPI:

```
Cliente (Next.js / Mobile)
        │
        ▼
┌───────────────────────────────────────────────┐
│              Routers (API REST)                │
│   auth.py · users.py · diagnose.py             │
└───────────────────┬───────────────────────────┘
                     │
        ┌────────────┼─────────────┐
        ▼            ▼             ▼
 ┌─────────────┐ ┌────────────┐ ┌──────────────────────┐
 │  Security   │ │ ML Service │ │   LLM Service         │
 │  (JWT/OAuth)│ │ CNN+GradCAM│ │   llm_service.py      │
 │             │ │            │ │   (LiteLLM · Fallback)│
 └─────────────┘ └────────────┘ └──────────────────────┘
        │            │                    │
        │            │         ┌──────────┴──────────┐
        │            │         ▼                     ▼
        │            │   Modelo primario       Fallback auto
        │            │  (cualquier proveedor) (DEFAULT_LLM_PROVIDER)
        ▼            ▼
 ┌─────────────────────────────────────────────┐
 │   PostgreSQL (SQLAlchemy 2.0 Async)          │
 │   S3 / Cloudflare R2 (Object Storage)        │
 └─────────────────────────────────────────────┘
```

**Decisiones de diseño clave:**

- **Asincronía de extremo a extremo**: SQLAlchemy 2.0 con `asyncpg`, LiteLLM en modo `acompletion` asíncrono, y `aioboto3` para el almacenamiento de objetos. Ninguna operación de I/O bloquea el *event loop*.
- **Carga única del modelo en RAM**: el modelo `.keras` se carga exactamente una vez durante el `lifespan` de FastAPI, evitando el costo de I/O y de reconstrucción del grafo de cómputo en cada request.
- **Abstracción Multi-LLM con LiteLLM**: una única interfaz unificada reemplaza la gestión de cinco SDKs propietarios. Añadir un nuevo proveedor es cuestión de configurar su API Key y pasar el identificador canónico correcto al endpoint.
- **Fallback automático**: si el proveedor seleccionado falla tras los reintentos internos, el sistema conmuta sin interrumpir el stream SSE hacia el modelo de respaldo configurado en `.env`.

---

## 🎯 Pilares Funcionales

### 1. Gestión de Usuarios y Seguridad
- Registro e inicio de sesión local (correo/contraseña) con hasheo **bcrypt** y emisión de **JWT**.
- Autenticación federada mediante **Google OAuth2** (verificación criptográfica de ID Tokens).
- Gestión de perfil (`/users/me`) y configuración de cuenta (`/users/me/settings`).

### 2. Persistencia e Historial Geoespacial
- Cada consulta se persiste en **PostgreSQL**, indexando: URL de la imagen (bucket S3/R2), veredicto probabilístico de la CNN, coordenadas geográficas exactas (`lat`, `lon`), marcas de tiempo y dispositivo de origen.

### 3. Explicabilidad Híbrida y Orquestación Multi-LLM
- Inferencia local de la CNN + cálculo de **Grad-CAM puro en TensorFlow** sobre la última capa convolucional, devolviendo una matriz de activación normalizada `[0.0, 1.0]`.
- Generación de reportes agronómicos en streaming (Server-Sent Events) a través de la capa **LiteLLM**, que enruta dinámicamente hacia **Gemini**, **OpenAI**, **Anthropic Claude**, **Groq** o **Azure OpenAI**, seleccionable por el cliente en tiempo de ejecución mediante un identificador de modelo.
- **Fallback automático con señalización SSE**: si el modelo primario falla, el sistema conmuta en caliente al proveedor de respaldo y notifica al cliente mediante un evento SSE dedicado (`event: fallback_activated`), sin cortar la conexión.

---

## 🧰 Stack Tecnológico

| Categoría | Tecnología |
|---|---|
| Framework Web | FastAPI + Uvicorn (ASGI) |
| Base de Datos | PostgreSQL 16 + SQLAlchemy 2.0 (async) + asyncpg |
| Machine Learning | TensorFlow-CPU 2.18 (CNN + Grad-CAM) |
| Procesamiento de Imágenes | Pillow + NumPy |
| Seguridad | passlib (bcrypt) + python-jose (JWT) + google-auth |
| Almacenamiento de Objetos | aioboto3 (S3 / Cloudflare R2) |
| **Abstracción LLM** | **LiteLLM 1.55** (interfaz unificada multi-proveedor) |
| Proveedores LLM | Gemini · OpenAI · Anthropic · Groq · Azure AI Inference |
| Contenedorización | Docker + Docker Compose |
| Validación de Datos | Pydantic v2 + pydantic-settings |

---

## 📁 Estructura del Proyecto

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Punto de entrada, lifespan, CORS, routers
│   ├── config.py                # Configuración centralizada (Pydantic Settings)
│   ├── database.py              # Motor asíncrono SQLAlchemy 2.0
│   ├── dependencies.py          # Dependencia get_current_user (JWT)
│   ├── routers/
│   │   ├── auth.py              # /auth/register, /auth/login, /auth/google
│   │   ├── users.py             # /users/me, /users/me/settings
│   │   └── diagnose.py          # /api/v1/diagnose, /stream-report (SSE + fallback)
│   ├── models/
│   │   └── database_models.py   # Entidades ORM (User, QueryHistory, etc.)
│   ├── schemas/
│   │   └── schemas.py           # DTOs Pydantic de entrada/salida
│   └── services/
│       ├── llm_service.py       # ★ Capa unificada LiteLLM + fallback automático
│       ├── llm_base.py          # BaseLLMService (contrato abstracto heredable)
│       ├── llm_providers.py     # Providers concretos con SDKs nativos (referencia)
│       ├── ml_service.py        # Inferencia CNN + Grad-CAM
│       ├── security_service.py  # Hashing, JWT, verificación Google OAuth2
│       └── storage_service.py   # Subida asíncrona a S3/R2
├── models/
│   ├── cnn_plantvillage.keras   # ⚠️ Debe ser provisto por el usuario
│   └── metadata_metrics.json
├── .env.sample
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

> **`llm_service.py`** es el punto de entrada exclusivo del router `diagnose.py` para toda la generación de texto. Los archivos `llm_base.py` y `llm_providers.py` se conservan como referencia de la implementación anterior con SDKs nativos.

---

## ✅ Requisitos Previos

- Docker Engine ≥ 24.x y Docker Compose ≥ 2.x
- El artefacto entrenado `cnn_plantvillage.keras` colocado en `backend/models/`
- Al menos una API Key válida de los proveedores LLM soportados
- (Opcional) Credenciales de Google OAuth2 y de un bucket S3/Cloudflare R2

---

## 🚀 Instalación y Despliegue

### 1. Clonar o descomprimir el proyecto

```bash
cd backend
```

### 2. Configurar variables de entorno

```bash
cp .env.sample .env
```

Edite `.env` y complete, como mínimo: `SECRET_KEY`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, y al menos una API Key de LLM. Defina `DEFAULT_LLM_PROVIDER` con el proveedor que actuará como fallback automático.

### 3. Colocar el modelo entrenado

```bash
cp /ruta/local/cnn_plantvillage.keras backend/models/
```

### 4. Levantar el stack orquestado

```bash
docker compose up --build
```

Docker Compose levantará primero el servicio `db` (PostgreSQL), esperará a que su `healthcheck` (`pg_isready`) reporte estado `healthy`, y solo entonces iniciará el servicio `web` (FastAPI), evitando condiciones de carrera en el arranque.

### 5. Verificar disponibilidad

```bash
curl http://localhost:8000/health
```

La documentación interactiva (Swagger UI) estará disponible en:

```
http://localhost:8000/docs
```

---

## 🔐 Variables de Entorno

| Variable | Descripción | Obligatoria |
|---|---|---|
| `SECRET_KEY` | Clave de firma de los JWT | Sí |
| `DATABASE_URL` | Cadena de conexión asíncrona a PostgreSQL | Sí |
| `GOOGLE_CLIENT_ID` | Client ID para validar tokens de Google OAuth2 | Para login con Google |
| `S3_ENDPOINT_URL` / `S3_BUCKET_NAME` / `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` | Credenciales del bucket de almacenamiento | Sí |
| `MODEL_PATH` | Ruta al artefacto `.keras` | Sí (default provisto) |
| `GEMINI_API_KEY` | Credencial de Google Gemini | Para usar ese proveedor |
| `OPENAI_API_KEY` | Credencial de OpenAI | Para usar ese proveedor |
| `ANTHROPIC_API_KEY` | Credencial de Anthropic Claude | Para usar ese proveedor |
| `GROQ_API_KEY` | Credencial de Groq | Para usar ese proveedor |
| `AZURE_INFERENCE_ENDPOINT` / `AZURE_INFERENCE_CREDENTIAL` | Credenciales de Azure AI Inference | Para usar ese proveedor |
| `DEFAULT_LLM_PROVIDER` | Modelo de respaldo para el fallback automático (alias o nombre canónico LiteLLM) | Sí (default: `gemini`) |
| `CORS_ORIGINS` | Orígenes permitidos (Next.js, app móvil), separados por coma | Sí |

> Consulte `.env.sample` para la lista completa con valores de ejemplo y documentación del formato aceptado por `DEFAULT_LLM_PROVIDER`.

---

## 🗄️ Modelo de Base de Datos

| Tabla | Descripción |
|---|---|
| `users` | UUID, email único, `hashed_password` (nulo si es Google), `auth_provider`, `full_name`, `is_active`, timestamps |
| `user_settings` | Preferencias por usuario: `preferred_llm`, `language`, `notifications_enabled` |
| `queries_history` | Historial geoespacial: `image_url`, `diagnostic_result`, `confidence_score`, `location_lat/lon`, `device_source`, `created_at` |
| `llm_interpretations` | Reportes generados por LLM, vinculados a una consulta (`query_id`), con el identificador del **modelo efectivo usado** (primario o fallback) y `generated_text` |

Las tablas se crean automáticamente al arrancar la aplicación (`init_models()` en el `lifespan`). Para entornos productivos críticos se recomienda migrar a **Alembic**.

---

## 📡 Documentación de la API

### Autenticación

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/auth/register` | Registro local (correo/contraseña) |
| `POST` | `/auth/login` | Inicio de sesión local |
| `POST` | `/auth/google` | Autenticación federada con Google OAuth2 |

### Usuarios

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/users/me` | Perfil del usuario autenticado |
| `PATCH` | `/users/me` | Actualiza datos del perfil |
| `GET` | `/users/me/settings` | Configuración de cuenta |
| `PATCH` | `/users/me/settings` | Actualiza preferencias (LLM, idioma, notificaciones) |

### Diagnóstico

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/api/v1/diagnose` | Recibe imagen + `lat`/`lon`, ejecuta CNN + Grad-CAM, persiste y retorna veredicto |
| `GET` | `/api/v1/diagnose/{query_id}/stream-report?model={identificador}` | Streaming SSE del reporte interpretativo con fallback automático |

> **Cambio de parámetro**: el query parameter del endpoint de streaming pasó de `?provider=` a `?model=` para reflejar que ahora acepta identificadores canónicos de LiteLLM además de alias cortos.

**Ejemplo de respuesta de `/api/v1/diagnose`:**

```json
{
  "query_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "diagnostic_result": "Tizon_Temprano",
  "confidence_score": 0.94,
  "probabilities": {
    "Sana": 0.03,
    "Tizon_Temprano": 0.94,
    "Tizon_Tardio": 0.03
  },
  "grad_cam": { "size": 7, "matrix": [[0.0, 0.12, 0.0]] },
  "image_url": "https://bucket.../diagnoses/uuid.jpg",
  "location_lat": -12.0464,
  "location_lon": -77.0428,
  "created_at": "2026-06-21T15:30:00Z"
}
```

Todos los endpoints de `/users` y `/api/v1/diagnose` requieren el header:

```
Authorization: Bearer <access_token>
```

---

## 🏭 Arquitectura Multi-LLM (LiteLLM + Fallback)

### Cómo funciona

`llm_service.py` es la única pieza de infraestructura que el router `diagnose.py` necesita conocer. Expone una función pública:

```python
async def stream_agronomic_report(
    model_name: str,        # alias corto o nombre canónico LiteLLM
    context: StreamReportContext | dict,
) -> AsyncGenerator[str, None]:
    ...
```

Internamente, la función delega en `litellm.acompletion(stream=True)`, que actúa como capa de traducción universal hacia el proveedor seleccionado:

```
stream_agronomic_report("groq/llama-3.3-70b-versatile", context)
        │
        ▼
   _try_stream()  →  litellm.acompletion(model="groq/...", stream=True)
        │                    └─ Groq SDK (transparente para el router)
        │
        │  ← si falla (RateLimitError / AuthenticationError / etc.)
        ▼
   _try_stream()  →  litellm.acompletion(model=_FALLBACK_MODEL, stream=True)
                             └─ proveedor configurado en DEFAULT_LLM_PROVIDER
```

### Identificadores de modelo aceptados

El parámetro `?model=` acepta dos formatos equivalentes:

| Alias corto | Identificador canónico LiteLLM | Proveedor |
|---|---|---|
| `gemini` | `gemini/gemini-2.0-flash` | Google Gemini |
| `gemini-pro` | `gemini/gemini-1.5-pro` | Google Gemini |
| `openai` | `openai/gpt-4o-mini` | OpenAI |
| `gpt4o` | `openai/gpt-4o` | OpenAI |
| `claude` | `anthropic/claude-sonnet-4-6` | Anthropic |
| `claude-opus` | `anthropic/claude-opus-4-6` | Anthropic |
| `groq` | `groq/llama-3.3-70b-versatile` | Groq |
| `azure` | `azure/<AZURE_DEPLOYMENT_NAME>` | Azure AI Inference |

### Política de reintentos y fallback

1. **Reintentos internos (LiteLLM)**: ante errores transitorios (429, 503, timeout), LiteLLM reintenta hasta 2 veces antes de propagar la excepción. Configurable via `_MAX_RETRIES` en `llm_service.py`.
2. **Fallback de proveedor**: si el modelo primario falla de forma definitiva, el sistema conmuta automáticamente hacia el modelo definido en `DEFAULT_LLM_PROVIDER`, emitiendo previamente el evento SSE `fallback_activated` con el detalle `primario→respaldo`.
3. **Error irrecuperable**: si tanto el primario como el fallback fallan, se emite el evento SSE `error` con la descripción y el generador termina limpiamente sin lanzar una excepción HTTP.

### Eventos SSE del endpoint `/stream-report`

```
data: <token>                      → fragmento de texto del reporte (N veces)

event: fallback_activated          → conmutación automática de proveedor
data: gemini/gemini-2.0-flash→groq/llama-3.3-70b-versatile

event: error                       → error irrecuperable (ambos modelos fallaron)
data: <descripción del error>

event: done                        → stream completado (con o sin fallback)
data: [DONE]
```

### Persistencia del modelo efectivo

Al finalizar el stream, el texto acumulado se persiste en `llm_interpretations` registrando el **identificador del modelo que generó el texto** (primario o fallback), permitiendo auditoría precisa de qué proveedor respondió en cada consulta histórica.

### Añadir un nuevo proveedor

Con LiteLLM basta con:
1. Añadir la API Key del nuevo proveedor al `.env`.
2. Incluir el alias en `_ALIAS_MAP` dentro de `llm_service.py` (opcional, solo si se quiere un alias corto).
3. Pasar el identificador canónico `proveedor/modelo` directamente al parámetro `?model=`.

No se requiere modificar el router, las schemas ni ningún otro archivo.

---

## 🔬 Explicabilidad: Grad-CAM

El algoritmo se implementa de forma íntegra en TensorFlow (sin librerías de XAI de terceros):

1. Se construye un `grad_model` con dos salidas: la activación de la última capa convolucional y los logits de salida.
2. Bajo `tf.GradientTape`, se calcula `∂(clase predicha)/∂(activación convolucional)`.
3. Se promedian los gradientes por canal (`alpha_k`), se realiza la combinación lineal ponderada de los mapas de activación, se aplica ReLU y se normaliza al rango `[0.0, 1.0]`.
4. El resultado se serializa como lista de listas (matriz `N x N`, típicamente `7x7` o `14x14`) lista para ser renderizada como mapa de calor en el frontend.

La última capa convolucional se resuelve **dinámicamente** recorriendo `model.layers` en reversa (configurable manualmente vía `LAST_CONV_LAYER_NAME` si la arquitectura lo requiere).

---

## 📦 Generación Automatizada del Proyecto

El script `generate_backend_zip.py` reconstruye toda la estructura de directorios y el contenido íntegro de cada archivo fuente, empaquetando el resultado en `backend_project.zip`:

```bash
python generate_backend_zip.py
```

Esto genera:
- `./backend/` — árbol completo de carpetas y archivos
- `./backend_project.zip` — el mismo árbol comprimido, listo para distribución

El script usa únicamente la librería estándar de Python (sin dependencias externas) y verifica, tras la generación, que los archivos estructuralmente críticos existan en disco antes de empaquetar.

---

## 🛠️ Troubleshooting

| Problema | Causa probable | Solución |
|---|---|---|
| `web` no inicia | `db` no superó el healthcheck | Revisar `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB` en `.env` |
| `event: error` en `/stream-report` con ambos modelos | API Keys del proveedor primario y del fallback no configuradas o inválidas | Completar `GEMINI_API_KEY` (u otro) y `DEFAULT_LLM_PROVIDER` en `.env` y reiniciar |
| `event: fallback_activated` inesperado | El proveedor primario superó su cuota o está caído | Normal: el sistema conmutó automáticamente. Revisar el estado del proveedor primario |
| El parámetro `?provider=` ya no funciona | El query parameter se renombró a `?model=` en esta versión | Usar `?model=gemini` (o el alias/identificador canónico deseado) |
| Error al cargar el modelo | `cnn_plantvillage.keras` ausente o ruta incorrecta | Verificar `MODEL_PATH` y la presencia física del archivo en `backend/models/` |
| `401 Unauthorized` en endpoints protegidos | Token JWT ausente, expirado o malformado | Reautenticar vía `/auth/login` y reenviar el header `Authorization: Bearer` |
| Imágenes no se suben | Credenciales S3/R2 incorrectas | Verificar `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` |

---

**Proyecto de Tesis — Ingeniería de Sistemas — Investigación en IA**
