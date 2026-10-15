import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------
# CONFIGURACIÓN DE LA APLICACIÓN
# -----------------------------------------

st.set_page_config(
    page_title="Clasificador CNN",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Clasificador de imágenes con CNN")
st.write(
    "Sube una imagen y la red neuronal convolucional "
    "intentará identificar a qué categoría pertenece."
)

# Categorías de CIFAR-10
class_names = [
    "Avión", "Auto", "Pájaro", "Gato", "Ciervo",
    "Perro", "Rana", "Caballo", "Barco", "Camión"
]

# Ruta del archivo de pesos
RUTA_PESOS = os.path.join(
    "modelo",
    "modelo_cnn_cifar10.weights.h5"
)

# -----------------------------------------
# CONSTRUIR Y CARGAR EL MODELO CNN
# -----------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(RUTA_PESOS):
        raise FileNotFoundError(
            f"No se encontró el archivo de pesos: {RUTA_PESOS}. "
            "Verifica que esté dentro de la carpeta modelo."
        )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),

        tf.keras.layers.Conv2D(
            32, (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            64, (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            64, (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # Cargar los pesos del entrenamiento
    model.load_weights(RUTA_PESOS)

    return model


try:
    model = load_model()
    st.success("Modelo CNN cargado correctamente.")
except Exception as e:
    st.error(f"No fue posible cargar el modelo: {e}")
    st.stop()

# -----------------------------------------
# CARGAR UNA IMAGEN
# -----------------------------------------

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------------
# REALIZAR LA PREDICCIÓN
# -----------------------------------------

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Imagen seleccionada",
            use_container_width=True
        )

        st.write("### Analizando imagen...")

        # CIFAR-10 utiliza imágenes de 32 x 32 píxeles
        image_resized = image.resize((32, 32))

        # Normalizar los píxeles entre 0 y 1
        img_array = np.array(
            image_resized,
            dtype=np.float32
        ) / 255.0

        # Añadir la dimensión del lote
        img_array = np.expand_dims(img_array, axis=0)

        # Obtener las probabilidades
        predictions = model.predict(img_array, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index]) * 100

        st.success(
            f"Predicción: {class_names[predicted_index]}"
        )

        st.info(
            f"Confianza estimada: {confidence:.2f}%"
        )

        # Mostrar las probabilidades de las diez categorías
        st.write("### Probabilidades por categoría")

        probabilities = {
            class_names[i]: float(predictions[i]) * 100
            for i in range(len(class_names))
        }

        st.bar_chart(probabilities)

        st.caption(
            "La confianza representa la probabilidad estimada por el modelo, "
            "no una garantía de que la clasificación sea correcta."
        )

    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
