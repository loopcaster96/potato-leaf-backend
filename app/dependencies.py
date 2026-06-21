"""
Dependencias inyectables compartidas a través de los routers de la API.

Centraliza la extracción y validación del usuario autenticado a partir
del esquema `Authorization: Bearer <token>`, desacoplando la lógica de
identidad de la implementación particular de cada endpoint protegido.
"""

import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.database_models import User
from app.services.security_service import decode_access_token

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(
    token: str = Depends(_oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Resuelve la entidad `User` autenticada a partir del JWT entrante.

    Decodifica el token para obtener el identificador del usuario
    (claim `sub`), lo consulta en PostgreSQL y verifica que la cuenta
    se encuentre activa. Lanza 401 si el usuario no existe o 403 si la
    cuenta ha sido desactivada administrativamente.
    """
    user_id_str = decode_access_token(token)

    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identificador de usuario inválido dentro del token.",
        ) from exc

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El usuario asociado a este token no existe.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="La cuenta de usuario se encuentra desactivada.",
        )

    return user
