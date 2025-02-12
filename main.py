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
import tkinter as tk
from tkinter import messagebox

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

def mostrar_imagenes_predicha(carpeta, etiqueta):
    """
    Busca en la carpeta 'carpeta' todas las imágenes cuyo nombre contenga la 'etiqueta'
    (de forma case-insensitive) y las muestra en una cuadrícula.
    Si no se encuentra ninguna imagen, se imprime un mensaje.
    """
    # Obtener todos los archivos cuyo nombre contenga la etiqueta (case-insensitive)
    archivos = [f for f in os.listdir(carpeta) if etiqueta.lower() in f.lower()]
    if not archivos:
        print(f"No se encontraron imágenes para la etiqueta '{etiqueta}'.")
        return
    
    # Configurar la figura (por ejemplo, en una sola fila)
    plt.figure(figsize=(4 * len(archivos), 4))
    for i, archivo in enumerate(archivos):
        ruta = os.path.join(carpeta, archivo)
        imagen = cv2.imread(ruta)
        if imagen is not None:
            # Convertir la imagen de BGR a RGB para mostrarla con matplotlib
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            plt.subplot(1, len(archivos), i + 1)
            plt.imshow(imagen_rgb)
            plt.title(archivo)
            plt.axis("off")
        else:
            print(f"No se pudo cargar la imagen: {ruta}")
    plt.tight_layout()
    plt.show()

def ask_question(question):
    """
    Muestra una ventana de diálogo con la pregunta y retorna True si el usuario
    responde 'Sí' y False si responde 'No'.
    """
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal
    respuesta = messagebox.askyesno("Confirmación", question)
    root.destroy()
    return respuesta

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
    #imagenes = preprocesamiento_imagen.cargar_imagenes()
    #imagenes_procesadas = preprocesamiento_imagen.procesar_y_guardar(imagenes)
    #preprocesamiento_imagen.procesar_y_guardar_binarizadas(imagenes_procesadas)
    #preprocesamiento_imagen.mostrar_imagenes(imagenes, num_por_clase=1)

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
    #procesador_kmeans.entrenar_y_evaluar()

    #procesador_kmeans.evaluar_precision_alternativo()

      # Paso 4: Predicción de nuevas imágenes
    print("\nPrediciendo nuevas imágenes...")
    procesador_kmeans.predecir_imagen_nueva(temp_folder=carpeta_imagenes_temp)

    #Interfaz gráfica: Preguntar si las imágenes predichas son correctas
    if not ask_question("¿Son correctas las etiqueta de las imágenes?"):
        print("Por favor, elimine las imágenes incorrectas de 'TempImagenes' y suba nuevas, luego ejecute nuevamente.")
        return

    # Paso 3:Procesamiento de audios, etc. (bloques comentados)
    extractor = FeatureExtractor(input_folder="AudiosProcesados", use_pca=True, n_components=30)
    print("\nProcesando audios de entrenamiento...")
    features_entrenamiento, labels, _ = extractor.procesar_todos_los_audios()
    # Visualizar las características en 3D (usando PCA)
    #extractor.visualizar_caracteristicas_3d_con_etiquetas()

    # Extraer nombres de archivos y etiquetas; se asume formato "procesado_[etiqueta]_..."
    file_names = [label[1] for label in labels]
    labels = [label[0] for label in labels]

    # Mostrar porcentaje de varianza retenida (si se usó PCA)
    #if extractor.use_pca and extractor.pca is not None:
        # Mostrar el scree plot con la varianza explicada y acumulada
        #extractor.mostrar_scree_plot()
    
    # ==============================
    # Paso 5: Entrenamiento Final y Evaluación en Todo el Conjunto
    clasificador = KnnAlgorithm(k=7)
    clasificador.fit(features_entrenamiento, labels)
    clasificador.save_model("knn_model.pkl")

    # ==============================
    # Paso 6: Procesamiento de Audio de Prueba y Predicción
    renombrar_archivo(carpeta_audios_temp)
    procesador_audio = AudioProcessor(input_folder=carpeta_audios_temp, output_folder=carpeta_audios_temp, silence_threshold=35)
    procesador_audio.eliminar_silencios("audio.ogg")
    archivo_procesado = os.path.join(carpeta_audios_temp, "procesado_audio.wav")
    
    if not os.path.exists(archivo_procesado):
        print("Error: No se procesó el audio correctamente.")
        return

    # Aseguramos que el flujo de extracción es el mismo que en entrenamiento
    extractor.input_folder = archivo_procesado
    _, _, features_prueba = extractor.procesar_todos_los_audios()
    prediccion = clasificador.predict(features_prueba)
    print(f"\nLa palabra predicha es: {prediccion}")

    # Preguntar al usuario mediante interfaz gráfica, incluyendo el nombre predicho
    if not ask_question(f"La verdura nombrada en el audio es: {prediccion} ¿Es correcta?"):
        print("Eliminando archivo de audio procesado. Por favor, suba uno nuevo en 'TempAudios'.")
        try:
            os.remove(archivo_procesado)
        except Exception as e:
            print(f"No se pudo eliminar el archivo: {e}")
        return



    # (Opcional) Mostrar imagen correspondiente a la predicción
    mostrar_imagenes_predicha(carpeta_imagenes_etiquetadas, prediccion)

    # (Opcional) Limpiar carpeta temporal
    # limpiar_carpeta(carpeta_audios_temp)

if __name__ == "__main__":
    main()
