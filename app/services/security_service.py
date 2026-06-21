"""
Servicio transversal de seguridad criptográfica y de identidad.

Centraliza tres responsabilidades sensibles del sistema:
1. Hasheo y verificación de contraseñas mediante bcrypt (vía passlib).
2. Emisión y decodificación de JSON Web Tokens (JWT) firmados con HS256.
3. Verificación criptográfica de los ID Tokens emitidos por Google OAuth2.

Aislar esta lógica en un módulo independiente evita la duplicación de
código sensible a través de los routers y facilita la auditoría de
seguridad como punto único de revisión.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    """Genera el hash bcrypt irreversible de una contraseña en texto plano."""
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica una contraseña en texto plano contra su hash bcrypt almacenado."""
    return _pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str) -> tuple[str, int]:
    """
    Genera un JWT firmado cuyo claim `sub` identifica al usuario.

    Retorna una tupla compuesta por el token codificado y su tiempo de
    expiración en segundos, requerido para construir la respuesta del
    endpoint de autenticación conforme al estándar OAuth2 Bearer.
    """
    expire_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expire_at = datetime.now(timezone.utc) + expire_delta
    payload: dict[str, Any] = {"sub": subject, "exp": expire_at}
    encoded_jwt = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt, int(expire_delta.total_seconds())


def decode_access_token(token: str) -> str:
    """
    Decodifica y valida un JWT, retornando el identificador del usuario
    (claim `sub`).

    Lanza una HTTPException 401 si el token es inválido, está corrupto
    o ha expirado, centralizando el manejo de errores de autenticación.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudo validar las credenciales de autenticación",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        subject: str | None = payload.get("sub")
        if subject is None:
            raise credentials_exception
        return subject
    except JWTError:
        raise credentials_exception


def verify_google_id_token(token: str) -> dict[str, Any]:
    """
    Verifica criptográficamente un ID Token de Google OAuth2 contra los
    servidores de Google, validando firma, emisor (`iss`) y audiencia
    (`aud`) frente al `GOOGLE_CLIENT_ID` configurado.

    Retorna el payload decodificado, que contiene como mínimo `email`,
    `name` y `sub` (identificador único de la cuenta de Google).
    """
    try:
        id_info = google_id_token.verify_oauth2_token(
            token, google_requests.Request(), audience=settings.GOOGLE_CLIENT_ID
        )
        return id_info
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token de Google OAuth2 inválido: {exc}",
        ) from exc
