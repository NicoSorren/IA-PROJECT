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

    #procesador = AudioProcessor()
    #procesador.procesar_todos_los_audios()
        
  
    # (Bloques relacionados con imágenes y KMeans quedan comentados, pues el foco es el audio)
    print("\nPrediciendo nuevas imágenes...")

    # ==============================
    # Paso 3: Procesamiento de Audios de Entrenamiento
    extractor = FeatureExtractor(input_folder="AudiosProcesados", use_pca=True, n_components=30)
    print("\nProcesando audios de entrenamiento...")
    features_entrenamiento, labels, _ = extractor.procesar_todos_los_audios()
    # Visualizar las características en 3D (usando PCA)
    extractor.visualizar_caracteristicas_3d_con_etiquetas()
    
    # Extraer nombres de archivos y etiquetas; se asume formato "procesado_[etiqueta]_..."
    file_names = [label[1] for label in labels]
    labels = [label[0] for label in labels]
    
    print("\nEstadísticas para la clase 'berenjena':")
    extractor.imprimir_estadisticas_por_clase("berenjena", num_componentes=3)

    print("\nEstadísticas para la clase 'zanahoria':")
    extractor.imprimir_estadisticas_por_clase("zanahoria", num_componentes=3)

    print("\nEstadísticas para la clase 'camote':")
    extractor.imprimir_estadisticas_por_clase("camote", num_componentes=3)

    print("\nEstadísticas para la clase 'papa':")
    extractor.imprimir_estadisticas_por_clase("papa", num_componentes=3)

    # Mostrar porcentaje de varianza retenida (si se usó PCA)
    if extractor.use_pca and extractor.pca is not None:
        #print("\nVarianza explicada por PCA:")
        #print(extractor.pca.explained_variance_ratio_)
        total_varianza = np.sum(extractor.pca.explained_variance_ratio_)
        #print("Suma total de varianza retenida: {:.2f}%".format(total_varianza * 100))

        # Mostrar el scree plot con la varianza explicada y acumulada
        extractor.mostrar_scree_plot()
    
    # ==============================
    # Paso 4: Validación Cruzada del Modelo KNN
    print("\nRealizando validación cruzada con K-Fold...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    labels_array = np.array(labels)  # Para facilitar el indexado

    for fold, (train_index, test_index) in enumerate(kf.split(features_entrenamiento)):
        # Extraer nombres de archivo para entrenamiento y test
        train_files = [file_names[i] for i in train_index]
        test_files = [file_names[i] for i in test_index]
        
        logging.debug(f"Fold {fold+1} - Archivos de entrenamiento: {train_files}")
        logging.debug(f"Fold {fold+1} - Archivos de prueba: {test_files}")
        
        X_train = features_entrenamiento[train_index]
        y_train = labels_array[train_index]
        X_test = features_entrenamiento[test_index]
        y_test = labels_array[test_index]
        
        knn_cv = KnnAlgorithm(k=7)
        knn_cv.fit(X_train, y_train)
        
        # Realizar predicciones para el conjunto de prueba de este fold
        y_pred = [knn_cv.predict(sample) for sample in X_test]
        accuracy_fold = np.sum(np.array(y_pred) == y_test) / len(y_test)
        accuracies.append(accuracy_fold)
        
        print(f"\nExactitud en Fold {fold+1}: {accuracy_fold * 100:.2f}%")
        logging.debug(f"Fold {fold+1} - Exactitud: {accuracy_fold * 100:.2f}%")
        
        print(f"\nMatriz de confusión para Fold {fold+1}:")
        conf_matrix = knn_cv.confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(conf_matrix)
        df_cm = df_cm.T
        print(df_cm)
        # Imprimir reporte de clasificación y matriz de confusión para este fold
        print(f"\nReporte de clasificación para Fold {fold+1}:")
        print(knn_cv.classification_report(y_test, y_pred))
        logging.debug(f"Fold {fold+1} - Reporte de clasificación:\n{knn_cv.classification_report(y_test, y_pred)}")
        logging.debug(f"Fold {fold+1} - Matriz de confusión:\n{knn_cv.confusion_matrix(y_test, y_pred)}")

    avg_accuracy = np.mean(accuracies)
    print(f"\nExactitud promedio en validación cruzada: {avg_accuracy * 100:.2f}%")
    logging.debug(f"Exactitud promedio en validación cruzada: {avg_accuracy * 100:.2f}%")
    
    # ==============================
    # Paso 5: Entrenamiento Final y Evaluación en Todo el Conjunto
    clasificador = KnnAlgorithm(k=7)
    clasificador.fit(features_entrenamiento, labels)
    clasificador.save_model("knn_model.pkl")
    print("\nEvaluación del modelo en el conjunto completo de entrenamiento:")
    clasificador.evaluate(features_entrenamiento, labels, file_names=file_names)

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

    # ==============================
    # Paso 7: Mostrar Imagen Correspondiente a la Predicción
    nombre_imagen = f"{prediccion}.jpg"
    mostrar_imagen_predicha(carpeta_imagenes_etiquetadas, nombre_imagen, prediccion)

    # (Opcional) Limpiar carpeta temporal
    # limpiar_carpeta(carpeta_audios_temp)

if __name__ == "__main__":
    main()
