import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from AudioProcessor import AudioProcessor
from FeatureExtractor import FeatureExtractor
from KnnAlgorithm import KnnAlgorithm
from sklearn.model_selection import KFold
from ImageProcessorKMeans import ImageProcessorKMeans
from ImageProcessor import ImageProcessor
import pandas as pd

# Configuración del log: escribe todos los mensajes de nivel DEBUG o superior en debug.log
logging.basicConfig(
    filename='debug.log',
    filemode='w',  # Sobrescribe el archivo en cada ejecución
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def limpiar_carpeta(carpeta):
    """Elimina todos los archivos dentro de una carpeta de forma segura."""
    if os.path.exists(carpeta):
        for archivo in os.listdir(carpeta):
            ruta_archivo = os.path.join(carpeta, archivo)
            try:
                os.remove(ruta_archivo)
            except Exception as e:
                print(f"No se pudo eliminar {ruta_archivo}: {e}")

def renombrar_archivo(carpeta, prefijo="WhatsApp", nuevo_nombre="audio.ogg"):
    """Renombra archivos en una carpeta que comienzan con un prefijo."""
    for archivo in os.listdir(carpeta):
        if archivo.endswith((".wav", ".ogg")):
            try:
                os.rename(os.path.join(carpeta, archivo), os.path.join(carpeta, nuevo_nombre))
                print(f"Renombrado: {archivo} -> {nuevo_nombre}")
                return os.path.join(carpeta, nuevo_nombre)
            except OSError as e:
                print(f"Error al renombrar {archivo}: {e}")
    return None

def mostrar_imagen_predicha(carpeta, nombre_imagen, prediccion):
    """Muestra la imagen predicha basada en el nombre."""
    ruta_imagen = os.path.join(carpeta, nombre_imagen)
    if os.path.exists(ruta_imagen):
        print(f"Mostrando imagen correspondiente a la predicción: {ruta_imagen}")
        imagen = cv2.imread(ruta_imagen)
        if imagen is not None:
            plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
            plt.title(f"Predicción: {prediccion}")
            plt.axis("off")
            plt.show()
        else:
            print(f"No se pudo cargar la imagen {ruta_imagen}")
    else:
        print(f"No se encontró la imagen etiquetada para la predicción '{prediccion}'.")

def main():
    # Configuración de carpetas
    carpeta_audios_temp = os.path.join(os.getcwd(), "TempAudios")
    carpeta_imagenes_temp = os.path.join(os.getcwd(), "TempImagenes")
    carpeta_imagenes_etiquetadas = os.path.join(os.getcwd(), "ImagenesEtiquetadas")
    carpeta_imagenes_verduras = os.path.join(os.getcwd(), "ImagenesVerduras")
    carpeta_imagenes_procesadas = os.path.join(os.getcwd(), "ImagenesProcesadas")
    carpeta_imagenes_segmentadas = os.path.join(os.getcwd(), "ImagenesSegmentadas")
    
    # Limpiar carpetas de imágenes segmentadas y etiquetadas para evitar acumulación de imágenes
    
    limpiar_carpeta(carpeta_imagenes_etiquetadas)

    # Paso 1: Procesamiento de Imágenes
    preprocesamiento_imagen = ImageProcessor(
        image_folder=carpeta_imagenes_verduras,
        processed_folder=carpeta_imagenes_procesadas
    )
    imagenes = preprocesamiento_imagen.cargar_imagenes()
    imagenes_procesadas = preprocesamiento_imagen.procesar_y_guardar(imagenes)
    
    #preprocesamiento_imagen.mostrar_imagenes(imagenes, num_por_clase=1)
    #preprocesamiento_imagen.procesar_y_guardar_binarizadas(imagenes_procesadas)
    #preprocesamiento_imagen.procesar_y_guardar_contornos_canny(num_por_clase=10)

    # Paso 2: Segmentación y Entrenamiento KMeans
    # Se especifica k=4 para la clasificación y, por ejemplo, k_segmentation=5 para la segmentación.
    procesador_kmeans = ImageProcessorKMeans(
        image_folder=carpeta_imagenes_procesadas,
        segmented_folder=carpeta_imagenes_segmentadas,
        k=4,
        k_segmentation=4
    )
    #print("\nProcesando y guardando segmentaciones...")
    #procesador_kmeans.procesar_y_guardar_segmentadas()
    #print("Entrenando modelo KMeans...")
    procesador_kmeans.entrenar_y_evaluar()

    procesador_kmeans.evaluar_precision_alternativo()

    # Paso 3: (Opcional) Procesamiento de audios, etc. (bloques comentados)

    # Paso 4: Predicción de nuevas imágenes
    print("\nPrediciendo nuevas imágenes...")
    procesador_kmeans.predecir_imagen_nueva(temp_folder=carpeta_imagenes_temp)

    # (Opcional) Mostrar imagen correspondiente a la predicción
    # mostrar_imagen_predicha(carpeta_imagenes_etiquetadas, "papa.jpg", "papa")

    # (Opcional) Limpiar carpeta temporal
    # limpiar_carpeta(carpeta_audios_temp)

if __name__ == "__main__":
    main()
