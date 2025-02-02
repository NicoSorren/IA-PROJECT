import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from AudioProcessor import AudioProcessor
from FeatureExtractor import FeatureExtractor
from KnnAlgorithm import KnnAlgorithm
from ImageProcessorKMeans import ImageProcessorKMeans
from ImageProcessor import ImageProcessor

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
  
    #procesador = AudioProcessor()
    #procesador.procesar_todos_los_audios()
        
    # Paso 1: Procesamiento de Imágenes
   # preprocesamiento_imagen = ImageProcessor(image_folder=carpeta_imagenes_verduras,
    #                                         processed_folder=carpeta_imagenes_procesadas)
   # imagenes = preprocesamiento_imagen.cargar_imagenes()
   # imagenes_procesadas = preprocesamiento_imagen.procesar_y_guardar(imagenes)
    #preprocesamiento_imagen.procesar_y_guardar_binarizadas(imagenes_procesadas)
    #preprocesamiento_imagen.moswtrar_imagenes(imagenes, num_por_clase=1)

    # Paso 2: Segmentación y Entrenamiento KMeans
    #procesador_kmeans = ImageProcessorKMeans(image_folder=carpeta_imagenes_procesadas)
    #print("\nProcesando y guardando segmentaciones...")
    #procesador_kmeans.procesar_y_guardar_segmentadas()
    #print("Entrenando modelo KMeans...")
    #procesador_kmeans.entrenar_y_evaluar()

    print("\nPrediciendo nuevas imágenes...")
    #procesador_kmeans.predecir_imagen_nueva(temp_folder=carpeta_imagenes_temp)

    # Paso 3: Procesamiento de Audios de Entrenamiento
    extractor = FeatureExtractor(input_folder="AudiosProcesados", use_pca=True, n_components=9)
    print("\nProcesando audios de entrenamiento...")
    features_entrenamiento, labels, _ = extractor.procesar_todos_los_audios()
    # Visualizar las características en 3D
    extractor.visualizar_caracteristicas_3d_con_etiquetas()
    file_names = [label[1] for label in labels]  # Extrae los nombres de archivo
    labels = [label[0] for label in labels]  # Solo deja las etiquetas


    clasificador = KnnAlgorithm(k=7)
    clasificador.fit(features_entrenamiento, labels)
    clasificador.save_model("knn_model.pkl")
    clasificador.evaluate(features_entrenamiento, labels, file_names=file_names)


    # Paso 4: Procesar Audio de Prueba
    renombrar_archivo(carpeta_audios_temp)
    procesador_audio = AudioProcessor(input_folder=carpeta_audios_temp, output_folder=carpeta_audios_temp, silence_threshold=35)
    procesador_audio.eliminar_silencios("audio.ogg")
    archivo_procesado = os.path.join(carpeta_audios_temp, "procesado_audio.wav")

    if not os.path.exists(archivo_procesado):
        print("Error: No se procesó el audio correctamente.")
        return

    # Paso 5: Predicción del Audio
    extractor.input_folder = archivo_procesado
    _, _, features_prueba = extractor.procesar_todos_los_audios()
    prediccion = clasificador.predict(features_prueba)
    print(f"\nLa palabra predicha es: {prediccion}")

    # Paso 6: Mostrar Imagen Correspondiente
    nombre_imagen = f"{prediccion}.jpg"
    mostrar_imagen_predicha(carpeta_imagenes_etiquetadas, nombre_imagen, prediccion)

    # Limpiar carpeta temporal
    #limpiar_carpeta(carpeta_audios_temp)

if __name__ == "__main__":
    main()
