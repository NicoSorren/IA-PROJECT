from AudioProcessor import AudioProcessor
from FeatureExtractor import FeatureExtractor
from KnnAlgorithm import KnnAlgorithm
import os
import numpy as np

def renombrar_archivo(carpeta, prefijo="WhatsApp", nuevo_nombre="audio.ogg"):
    """Renombra archivos en una carpeta que comienzan con un prefijo."""
    for archivo in os.listdir(carpeta):
        if archivo.startswith(prefijo) and archivo.endswith(".ogg"):
            try:
                os.rename(os.path.join(carpeta, archivo), os.path.join(carpeta, nuevo_nombre))
                print(f"Renombrado: {archivo} -> {nuevo_nombre}")
                return os.path.join(carpeta, nuevo_nombre)
            except OSError as e:
                print(f"Error al renombrar {archivo}: {e}")
    return None

def main():
    # Paso 1: Procesar audios de entrenamiento en "AudiosProcesados"
    renombrar_archivo("TempAudios")
    extractor = FeatureExtractor(input_folder="AudiosProcesados", use_pca=True, n_components=9)
    print("\nProcesando audios de entrenamiento...")
    features_entrenamiento, labels, _ = extractor.procesar_todos_los_audios()

    # Mostrar características extraídas de los audios procesados
    print("\nCaracterísticas extraídas de los audios de entrenamiento:")
    for feature, label in zip(features_entrenamiento, labels):
        print(f"{label}: {feature}")

    # Paso 2: Entrenar el modelo KNN con los audios de entrenamiento
    clasificador = KnnAlgorithm(k=4)  # Instancia del modelo KNN
    clasificador.fit(features_entrenamiento, labels)  # Entrenar modelo con las características extraídas

    # Guardar el modelo entrenado
    clasificador.save_model("knn_model.pkl")
    clasificador.evaluate(features_entrenamiento, labels)  # Evaluamos con los mismos datos de entrenamiento

    # Paso 3: Procesar el archivo de audio de prueba 'papa_prueba.ogg' utilizando AudioProcessor
    archivo_audio = "audio.ogg"
    procesador = AudioProcessor(input_folder="TempAudios", output_folder="TempAudios")
    procesador.eliminar_silencios(archivo_audio)  # Procesar el archivo y almacenarlo en TempAudios como .wav

    # Verificar si el archivo procesado está disponible
    archivo_procesado = "TempAudios/procesado_audio.wav"
    if os.path.exists(archivo_procesado):
        print(f"El archivo procesado se ha guardado correctamente como {archivo_procesado}")
    else:
        print(f"Error: El archivo procesado {archivo_procesado} no se ha guardado correctamente.")
        return

    # Paso 4: Extraer características del archivo procesado utilizando el extractor ajustado
    print("\nProcesando el archivo de prueba...")
    extractor.input_folder = archivo_procesado  # Cambiamos el input_folder al archivo de prueba
    _, _, features_prueba = extractor.procesar_todos_los_audios()  # Extraemos las características
    extractor.visualizar_caracteristicas_3d_con_etiquetas()

    print("\nCaracterísticas del archivo de prueba transformadas por PCA:")

    # Paso 5: Usar el modelo entrenado KNN para predecir la palabra
    prediccion = clasificador.predict(features_prueba)  # Usamos las características del archivo de prueba
    print(f"\nLa palabra predicha es: {prediccion}")

if __name__ == "__main__":
    main()