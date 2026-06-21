"""
Router de gestión del perfil de usuario y de sus preferencias de cuenta.

Todos los endpoints expuestos en este módulo están protegidos mediante
la dependencia `get_current_user`, que exige la presencia de un JWT
válido en el encabezado `Authorization: Bearer <token>`.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.database_models import QueryHistory, User, UserSettings
from app.schemas.schemas import (
    QueryHistoryItemResponse,
    UserProfileResponse,
    UserProfileUpdateRequest,
    UserSettingsResponse,
    UserSettingsUpdateRequest,
)

router = APIRouter(prefix="/users", tags=["Usuarios"])


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Obtiene el perfil del usuario autenticado",
)
async def read_current_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Retorna el perfil completo del usuario asociado al JWT entrante."""
    return current_user


@router.patch(
    "/me",
    response_model=UserProfileResponse,
    summary="Actualiza los datos básicos del perfil del usuario",
)
async def update_current_user(
    payload: UserProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Aplica una actualización parcial sobre los campos editables del
    perfil del usuario (actualmente, el nombre completo).
    """
    if payload.full_name is not None:
        current_user.full_name = payload.full_name

    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.get(
    "/me/settings",
    response_model=UserSettingsResponse,
    summary="Obtiene la configuración de cuenta del usuario autenticado",
)
async def read_current_user_settings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserSettings:
    """Retorna la configuración personalizada asociada al usuario actual."""
    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    user_settings = result.scalar_one_or_none()

    if user_settings is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se encontró configuración asociada a este usuario.",
        )
    return user_settings


@router.patch(
    "/me/settings",
    response_model=UserSettingsResponse,
    summary="Actualiza la configuración de cuenta del usuario autenticado",
)
async def update_current_user_settings(
    payload: UserSettingsUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserSettings:
    """
    Aplica una actualización parcial sobre las preferencias del usuario:
    proveedor de LLM por defecto, idioma de interfaz y habilitación de
    notificaciones. Solo se modifican los campos explícitamente
    presentes en el payload, preservando el resto de valores actuales.
    """
    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    user_settings = result.scalar_one_or_none()

    if user_settings is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se encontró configuración asociada a este usuario.",
        )

    update_data = payload.model_dump(exclude_unset=True)
    for field_name, field_value in update_data.items():
        setattr(user_settings, field_name, field_value)

    db.add(user_settings)
    await db.commit()
    await db.refresh(user_settings)
    return user_settings


@router.get(
    "/me/history",
    response_model=list[QueryHistoryItemResponse],
    summary="Obtiene el historial de diagnósticos del usuario autenticado",
)
async def read_current_user_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[QueryHistory]:
    """
    Retorna la lista de diagnósticos realizados por el usuario actual,
    ordenados descendentemente por fecha de creación (los más recientes primero).
    Incluye parámetros de paginación opcionales (limit y offset).
    """
    result = await db.execute(
        select(QueryHistory)
        .where(QueryHistory.user_id == current_user.id)
        .order_by(QueryHistory.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())
