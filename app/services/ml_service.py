"""
Servicio de inferencia local de Machine Learning y de Explicabilidad
(Explainable AI, XAI) mediante el algoritmo Grad-CAM.

Este módulo encapsula la totalidad del ciclo de vida del modelo CNN:
carga única en memoria al arrancar el proceso (gestionada externamente
por el `lifespan` de FastAPI), preprocesamiento determinista de las
imágenes de entrada, inferencia probabilística multiclase, y el cálculo
puro en TensorFlow del mapa de calor Grad-CAM sobre la última capa
convolucional, sin dependencias de librerías de terceros externas a
TensorFlow para el cálculo de gradientes.
"""

import io
import logging

import numpy as np
import tensorflow as tf
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class MLInferenceService:
    """
    Encapsula el modelo CNN cargado en RAM y expone los métodos de
    preprocesamiento, inferencia y generación de explicabilidad Grad-CAM.

    La instancia de esta clase debe ser única durante todo el ciclo de
    vida del proceso (patrón Singleton gestionado vía `app.state` en el
    `lifespan` de FastAPI), evitando el costo de I/O y de inicialización
    de grafos de cómputo en cada request entrante.
    """

    def __init__(self, model_path: str, class_names: list[str], input_size: int):
        self.model_path = model_path
        self.class_names = class_names
        self.input_size = input_size
        self.model: tf.keras.Model | None = None
        self.last_conv_layer_name: str | None = None

    def load_model(self) -> None:
        """
        Carga el artefacto `.keras` en memoria y construye el sub-modelo
        de gradientes (`grad_model`) requerido por Grad-CAM.

        El sub-modelo de gradientes expone simultáneamente la activación
        de la última capa convolucional y la salida final de la red,
        permitiendo calcular `d(salida)/d(activación)` con una sola
        pasada hacia adelante bajo el contexto de `tf.GradientTape`.
        """
        logger.info("Cargando modelo CNN desde: %s", self.model_path)
        
        # Guardar constructor original de la clase base Layer
        original_layer_init = tf.keras.layers.Layer.__init__
        
        def safe_layer_init(self_layer, *args, **kwargs):
            # Eliminar argumentos obsoletos de Keras 2
            kwargs.pop("renorm", None)
            kwargs.pop("renorm_clipping", None)
            kwargs.pop("renorm_momentum", None)
            kwargs.pop("quantization_config", None)
            original_layer_init(self_layer, *args, **kwargs)
            
        try:
            # Parchear temporalmente la clase maestra Layer
            tf.keras.layers.Layer.__init__ = safe_layer_init
            self.model = tf.keras.models.load_model(self.model_path)
        finally:
            # Restaurar estado original
            tf.keras.layers.Layer.__init__ = original_layer_init

        self.last_conv_layer_name = self._resolve_last_conv_layer()

        logger.info(
            "Modelo cargado exitosamente. Última capa convolucional detectada: %s",
            self.last_conv_layer_name,
        )

    def _resolve_last_conv_layer(self) -> str:
        """
        Determina dinámicamente el nombre de la última capa convolucional
        del modelo cargado, recorriendo la pila de capas en orden inverso.

        Si `settings.LAST_CONV_LAYER_NAME` ha sido fijado explícitamente
        a un valor distinto de `"auto"`, se respeta dicha configuración
        manual, lo cual es útil cuando la arquitectura posee bloques
        residuales o ramas múltiples donde la heurística automática
        podría no ser unívoca.
        """
        if settings.LAST_CONV_LAYER_NAME and settings.LAST_CONV_LAYER_NAME != "auto":
            return settings.LAST_CONV_LAYER_NAME

        assert self.model is not None
        for layer in reversed(self.model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
                return layer.name

        raise ValueError(
            "No se encontró ninguna capa convolucional (Conv2D) en el modelo "
            "cargado. Verifique la arquitectura de cnn_plantvillage.keras."
        )

    def preprocess_image(self, image_bytes: bytes) -> tf.Tensor:
        """
        Aplica el pipeline de preprocesamiento determinista requerido por
        la CNN: decodificación, conversión a RGB, redimensionado a
        224x224x3 y normalización de píxeles al rango [0.0, 1.0].

        Se utiliza Pillow para la decodificación robusta de formatos de
        imagen heterogéneos (JPEG, PNG, WEBP) provenientes de cámaras
        móviles y navegadores web antes de delegar el resto del pipeline
        numérico a TensorFlow.
        """
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_image = pil_image.resize(
            (self.input_size, self.input_size), Image.Resampling.BILINEAR
        )
        image_array = np.asarray(pil_image, dtype=np.float32)
        # NOTA: No se divide por 255.0. Las arquitecturas modernas en Keras
        # (como EfficientNet) incluyen internamente su propia capa de Rescaling.
        image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        return image_tensor

    def predict(self, image_tensor: tf.Tensor) -> tuple[str, float, dict[str, float]]:
        """
        Ejecuta la inferencia probabilística multiclase sobre el tensor
        preprocesado.

        Retorna una tupla con: la etiqueta de la clase ganadora, el
        score de confianza asociado (softmax máximo), y el diccionario
        completo de probabilidades por clase para trazabilidad y
        auditoría del veredicto.
        """
        if self.model is None:
            raise RuntimeError("El modelo CNN no ha sido cargado en memoria.")

        predictions = self.model(image_tensor, training=False)
        probabilities = tf.nn.softmax(predictions[0]).numpy()

        predicted_index = int(np.argmax(probabilities))
        predicted_label = self.class_names[predicted_index]
        confidence_score = float(probabilities[predicted_index])

        probability_map = {
            class_name: float(probabilities[idx])
            for idx, class_name in enumerate(self.class_names)
        }
        return predicted_label, confidence_score, probability_map

    def compute_grad_cam(
        self, image_tensor: tf.Tensor, predicted_index: int
    ) -> list[list[float]]:
        """
        Calcula el mapa de activación Grad-CAM puro en TensorFlow para la
        clase predicha, interceptando los gradientes de la última capa
        convolucional respecto al logit de salida correspondiente.

        El algoritmo sigue la formulación original de Selvaraju et al.
        (2017): se obtiene el promedio espacial de los gradientes por
        canal (pesos de importancia `alpha_k`), se realiza la combinación
        lineal ponderada de los mapas de activación, se aplica ReLU para
        retener únicamente las contribuciones positivas, y finalmente se
        normaliza el resultado al rango [0.0, 1.0] para su consumo directo
        por el frontend como una superposición de calor (heatmap).
        """
        if self.model is None or self.last_conv_layer_name is None:
            raise RuntimeError("El modelo no ha sido inicializado.")

        # Construir un sub-modelo que exponga simultáneamente la salida
        # de la capa convolucional y la predicción final.
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output,
            ],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_tensor, training=False)
            class_channel = predictions[:, predicted_index]

        gradients = tape.gradient(class_channel, conv_outputs)

        if gradients is None:
            raise RuntimeError(
                "No fue posible calcular los gradientes de Grad-CAM. "
                "Verifique que la capa convolucional seleccionada participe "
                "en el grafo de cómputo hacia la salida."
            )

        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(
            tf.multiply(pooled_gradients, conv_outputs), axis=-1
        )
        heatmap = tf.maximum(heatmap, 0)

        max_value = tf.reduce_max(heatmap)
        if max_value > 0:
            heatmap = heatmap / max_value

        heatmap_array = heatmap.numpy().astype(float)
        return [[round(float(value), 6) for value in row] for row in heatmap_array]


def build_ml_service() -> MLInferenceService:
    """
    Factory de construcción del servicio de inferencia, leyendo la
    configuración global de la aplicación.

    Se invoca exactamente una vez durante el evento `startup` del
    `lifespan` de FastAPI, y la instancia resultante se almacena en
    `app.state.ml_service` para su reutilización transversal en los
    routers de diagnóstico.
    """
    return MLInferenceService(
        model_path=settings.MODEL_PATH,
        class_names=settings.class_names_list,
        input_size=settings.MODEL_INPUT_SIZE,
    )
