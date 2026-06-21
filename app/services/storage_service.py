"""
Servicio de almacenamiento de objetos compatible con S3 y Cloudflare R2.

Utiliza `aioboto3` para realizar la subida de imágenes de forma
completamente asíncrona, evitando bloquear el event loop de FastAPI
durante operaciones de I/O de red hacia el bucket. La compatibilidad
con Cloudflare R2 se logra mediante la configuración de un
`endpoint_url` alternativo, ya que R2 implementa la API S3 de forma
nativa.
"""

import logging
import uuid

import aioboto3

from app.config import settings

logger = logging.getLogger(__name__)


class ObjectStorageService:
    """
    Encapsula la interacción asíncrona con el bucket de almacenamiento
    de objetos configurado (AWS S3 o Cloudflare R2), exponiendo un
    método de alto nivel para la subida de imágenes de diagnóstico.
    """

    def __init__(
        self,
        bucket_name: str,
        endpoint_url: str | None,
        public_url: str | None,
        access_key_id: str | None,
        secret_access_key: str | None,
        region: str,
    ):
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.public_url = public_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region
        self._session = aioboto3.Session()

    async def upload_image(self, file_bytes: bytes, content_type: str, user_id: str) -> str:
        """
        Sube los bytes de la imagen al bucket configurado bajo una clave
        única generada con UUID4, organizada dentro de la carpeta del usuario.

        Retorna la URL pública (o el endpoint configurado) bajo la cual
        el recurso queda accesible para su posterior renderizado en el
        frontend web y móvil.
        """
        object_key = f"users/{user_id}/diagnoses/{uuid.uuid4()}.jpg"

        async with self._session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region,
        ) as s3_client:
            await s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=file_bytes,
                ContentType=content_type,
            )

        if self.public_url:
            return f"{self.public_url.rstrip('/')}/{object_key}"
            
        base_url = self.endpoint_url or f"https://{self.bucket_name}.s3.amazonaws.com"
        return f"{base_url.rstrip('/')}/{self.bucket_name}/{object_key}"


def build_storage_service() -> ObjectStorageService:
    """
    Factory de construcción del servicio de almacenamiento de objetos,
    leyendo la configuración global de credenciales y endpoint.
    """
    return ObjectStorageService(
        bucket_name=settings.S3_BUCKET_NAME,
        endpoint_url=settings.S3_ENDPOINT_URL,
        public_url=settings.S3_PUBLIC_URL,
        access_key_id=settings.S3_ACCESS_KEY_ID,
        secret_access_key=settings.S3_SECRET_ACCESS_KEY,
        region=settings.S3_REGION,
    )
