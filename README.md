\# Clasificación de imágenes con CNN y Streamlit



\## 1. Descripción del proyecto



Este proyecto implementa una Red Neuronal Convolucional (CNN) para clasificar imágenes utilizando el conjunto de datos CIFAR-10. Además, incorpora una aplicación web desarrollada con Streamlit que permite cargar una imagen y obtener una predicción de su categoría junto con la confianza estimada por el modelo.



\## 2. Objetivo



Desarrollar un modelo de Deep Learning capaz de clasificar imágenes en diez categorías y permitir su utilización mediante una interfaz web sencilla.



\## 3. Dataset



Se utiliza CIFAR-10, un conjunto de datos que contiene imágenes de 32 × 32 píxeles distribuidas en diez categorías:



\* Avión

\* Automóvil

\* Pájaro

\* Gato

\* Ciervo

\* Perro

\* Rana

\* Caballo

\* Barco

\* Camión



El conjunto de datos proporciona imágenes para entrenamiento y prueba. Una parte de los datos de entrenamiento se reserva para validación.



\## 4. Tecnologías utilizadas



\* Python

\* TensorFlow y Keras

\* NumPy

\* Matplotlib

\* Scikit-learn

\* Pillow

\* Streamlit

\* Google Colab



\## 5. Arquitectura del modelo



La CNN utiliza las siguientes capas:



1\. Capa de entrada para imágenes RGB de 32 × 32 píxeles.

2\. Capa convolucional con 32 filtros.

3\. Capa de agrupamiento máximo.

4\. Capa convolucional con 64 filtros.

5\. Capa de agrupamiento máximo.

6\. Segunda capa convolucional con 64 filtros.

7\. Capa Flatten para convertir las características en un vector.

8\. Capa densa con 64 neuronas.

9\. Capa de salida con 10 neuronas y activación softmax.



El modelo se configura con el optimizador Adam y la función de pérdida sparse categorical crossentropy.



\## 6. Estructura del proyecto



```text

Proyecto\_CNN/

├── Entrenamiento\_CNN\_CIFAR10.ipynb

├── app.py

├── requirements.txt

├── README.md

├── modelo/

│   └── modelo\_cnn\_cifar10.weights.h5

└── resultados/

&#x20;   ├── entrenamiento.png

&#x20;   ├── perdida.png

&#x20;   ├── matriz\_confusion.png

&#x20;   └── 10\_predicciones.png

```



\## 7. Instalación



Se recomienda utilizar un entorno virtual de Python.



Instalar las dependencias con:



```bash

pip install -r requirements.txt

```



\## 8. Ejecución de la aplicación



Desde la carpeta principal del proyecto, ejecutar:



```bash

streamlit run app.py

```



La aplicación abrirá una interfaz web donde se puede seleccionar una imagen en formato JPG, JPEG o PNG.



El sistema redimensiona la imagen a 32 × 32 píxeles, normaliza sus valores y utiliza la CNN entrenada para generar la predicción.



\## 9. Resultados



La carpeta `resultados` contiene las gráficas de precisión y pérdida del entrenamiento, la matriz de confusión y una visualización de diez predicciones realizadas sobre imágenes de prueba.



Las métricas obtenidas deben consultarse en los resultados generados por el notebook.



\## 10. Limitaciones



El modelo fue entrenado con CIFAR-10, cuyas imágenes son pequeñas y pertenecen a diez categorías específicas. Por esta razón, las fotografías externas pueden producir predicciones incorrectas, especialmente cuando presentan fondos, objetos o condiciones visuales diferentes a los datos de entrenamiento.



\## 11. Autoría



Proyecto académico de Deep Learning sobre clasificación de imágenes con CNN y Streamlit.



