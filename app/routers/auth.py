"""
Router de autenticación y gestión de identidad.

Expone los tres flujos de entrada soportados por el sistema:
registro tradicional, inicio de sesión tradicional, y autenticación
federada mediante Google OAuth2. Todos los flujos convergen en la
emisión de un JWT homogéneo, de modo que el resto de la API permanezca
agnóstico al mecanismo de autenticación original del usuario.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.database_models import AuthProviderEnum, User, UserSettings
from app.schemas.schemas import (
    GoogleAuthRequest,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
)
from app.services.security_service import (
    create_access_token,
    hash_password,
    verify_google_id_token,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["Autenticación"])


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Registro de usuario mediante correo y contraseña",
)
async def register(
    payload: UserRegisterRequest, db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Registra un nuevo usuario bajo el flujo de autenticación local.

    Verifica unicidad del correo electrónico, aplica hasheo bcrypt
    irreversible sobre la contraseña en texto plano, y crea de forma
    transaccional el registro de `User` junto con su `UserSettings`
    asociado, inicializado con valores por defecto.
    """
    existing_user = await db.execute(select(User).where(User.email == payload.email))
    if existing_user.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe una cuenta registrada con este correo electrónico.",
        )

    new_user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        auth_provider=AuthProviderEnum.LOCAL,
        full_name=payload.full_name,
        is_active=True,
    )
    db.add(new_user)
    await db.flush()

    new_settings = UserSettings(user_id=new_user.id)
    db.add(new_settings)
    await db.commit()

    access_token, expires_in = create_access_token(subject=str(new_user.id))
    return TokenResponse(access_token=access_token, expires_in=expires_in)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Inicio de sesión mediante correo y contraseña",
)
async def login(
    payload: UserLoginRequest, db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Autentica a un usuario existente bajo el flujo local.

    Verifica que la cuenta haya sido registrada mediante `auth_provider
    = local` (rechazando intentos de login con contraseña sobre cuentas
    federadas con Google) y compara el hash bcrypt almacenado contra la
    contraseña en texto plano recibida.
    """
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    if (
        user is None
        or user.auth_provider != AuthProviderEnum.LOCAL
        or user.hashed_password is None
        or not verify_password(payload.password, user.hashed_password)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo electrónico o contraseña incorrectos.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="La cuenta de usuario se encuentra desactivada.",
        )

    access_token, expires_in = create_access_token(subject=str(user.id))
    return TokenResponse(access_token=access_token, expires_in=expires_in)


@router.post(
    "/google",
    response_model=TokenResponse,
    summary="Autenticación federada mediante Google OAuth2",
)
async def google_auth(
    payload: GoogleAuthRequest, db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Autentica (o registra de forma transparente, patrón "sign-up on
    first sign-in") a un usuario mediante un ID Token de Google OAuth2.

    Verifica la firma criptográfica del token contra los servidores de
    Google, extrae el correo electrónico verificado y crea la cuenta
    local si esta es la primera vez que el usuario inicia sesión,
    marcando `auth_provider = google` y dejando `hashed_password` nulo.
    """
    id_info = verify_google_id_token(payload.id_token)
    email = id_info.get("email")
    full_name = id_info.get("name")

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El token de Google no contiene un correo electrónico válido.",
        )

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            email=email,
            hashed_password=None,
            auth_provider=AuthProviderEnum.GOOGLE,
            full_name=full_name,
            is_active=True,
        )
        db.add(user)
        await db.flush()

        new_settings = UserSettings(user_id=user.id)
        db.add(new_settings)
        await db.commit()
    elif user.auth_provider != AuthProviderEnum.GOOGLE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Este correo electrónico ya está registrado mediante el "
                "flujo de autenticación local."
            ),
        )

    access_token, expires_in = create_access_token(subject=str(user.id))
    return TokenResponse(access_token=access_token, expires_in=expires_in)
