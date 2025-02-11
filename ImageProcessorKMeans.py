from KMeansAlgorithm import KMeansManual
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Para el gráfico 3D
from ImageProcessor import ImageProcessor
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

class ImageProcessorKMeans:
    def __init__(self, image_folder="ImagenesProcesadas", segmented_folder="ImagenesSegmentadas", k=4, k_segmentation=None):
        """
        Parámetros:
            image_folder: Carpeta con las imágenes originales procesadas.
            segmented_folder: Carpeta donde se guardarán las imágenes segmentadas.
            k: Número de clusters para la clasificación (y normalización de características).
            k_segmentation: Número de clusters a utilizar en la segmentación de la imagen.
                           Si no se especifica, se usará el mismo valor que k.
        """
        self.image_folder = image_folder
        self.segmented_folder = segmented_folder
        self.binarized_folder = "ImagenesBinarizadas"
        self.k = k
        # Si k_segmentation no se especifica, se toma k; de lo contrario se usa el valor proporcionado.
        self.k_segmentation = k_segmentation if k_segmentation is not None else k
        # La instancia de KMeans para clasificación usa k (número de clases)
        self.kmeans = KMeansManual(n_clusters=self.k, max_iter=100, tol=1e-4)
        self.image_processor = ImageProcessor()
        os.makedirs(self.segmented_folder, exist_ok=True)

    def aplicar_kmeans(self, imagen):
        """
        Aplica K-Means Manual para segmentar la imagen en k_segmentation clusters.
        Se utiliza un valor de k_segmentation distinto para la segmentación si se especificó.
        """
        original_shape = imagen.shape
        imagen_reshape = imagen.reshape((-1, 3))
        # Crear una instancia separada de KMeans para segmentación
        kmeans_segmentation = KMeansManual(n_clusters=self.k_segmentation, max_iter=100, tol=1e-4)
        kmeans_segmentation.fit(imagen_reshape)
        labels = kmeans_segmentation.predict(imagen_reshape)
        colores_centrales = kmeans_segmentation.centroides
        imagen_segmentada = colores_centrales[labels].reshape(original_shape)
        return imagen_segmentada.astype(np.uint8)

    def procesar_y_guardar_segmentadas(self):
        """
        Aplica K-Means a todas las imágenes y las guarda.
        Las imágenes se segmentan utilizando k_segmentation clusters.
        """
        for verdura in os.listdir(self.image_folder):
            ruta_verdura = os.path.join(self.image_folder, verdura)
            if os.path.isdir(ruta_verdura):
                carpeta_destino = os.path.join(self.segmented_folder, verdura)
                os.makedirs(carpeta_destino, exist_ok=True)
                for imagen_nombre in os.listdir(ruta_verdura):
                    ruta_imagen = os.path.join(ruta_verdura, imagen_nombre)
                    imagen = cv2.imread(ruta_imagen)
                    if imagen is not None:
                        # Usamos la segmentación con k_segmentation
                        imagen_segmentada = self.aplicar_kmeans(imagen)
                        ruta_guardado = os.path.join(carpeta_destino, f"segmentada_{imagen_nombre}")
                        cv2.imwrite(ruta_guardado, cv2.cvtColor(imagen_segmentada, cv2.COLOR_RGB2BGR))
                        #print(f"Imagen segmentada guardada: {ruta_guardado}")

    def extraer_caracteristicas_color(self, folder):
        """
        Extrae las características promedio RGB de las imágenes en una carpeta específica.
        Se asume que el nombre de la carpeta (o del archivo) indica la etiqueta real.
        """
        caracteristicas = []
        etiquetas = []
        for verdura in os.listdir(folder):
            ruta_verdura = os.path.join(folder, verdura)
            if os.path.isdir(ruta_verdura):
                for imagen_nombre in os.listdir(ruta_verdura):
                    ruta_imagen = os.path.join(ruta_verdura, imagen_nombre)
                    imagen = cv2.imread(ruta_imagen)
                    if imagen is not None:
                        promedio_color = np.mean(imagen, axis=(0, 1))  # Promedio RGB
                        caracteristicas.append(promedio_color)
                        etiquetas.append(verdura)  # Se extrae la etiqueta según la carpeta
        return np.array(caracteristicas), np.array(etiquetas)

    def extraer_caracteristicas_forma(self, input_folder="ImagenesContorno"):
        """
        Recorre todas las imágenes en la carpeta 'input_folder' (por defecto, ImagenesContorno)
        y extrae dos características de forma para el contorno principal de cada imagen:
            - Redondez (Circularidad)
            - Alargamiento (relación entre eje mayor y eje menor de la elipse ajustada)
            
        Retorna un diccionario con la siguiente estructura:
            {
              'nombre_clase': [
                  (nombre_imagen, redondez, alargamiento),
                  (nombre_imagen, redondez, alargamiento),
                  ...
              ],
              ...
            }
        """
        caracteristicas_forma = {}
        
        # Listar las subcarpetas (clases) en input_folder
        clases = os.listdir(input_folder)
        
        for clase in clases:
            carpeta_clase = os.path.join(input_folder, clase)
            if not os.path.isdir(carpeta_clase):
                continue
            
            caracteristicas_forma[clase] = []
            imagenes = os.listdir(carpeta_clase)
            
            for nombre in imagenes:
                ruta_imagen = os.path.join(carpeta_clase, nombre)
                # Leer la imagen; se espera que esté en un formato adecuado (por ejemplo, en BGR)
                img = cv2.imread(ruta_imagen)
                if img is None:
                    print(f"No se pudo leer la imagen: {ruta_imagen}")
                    continue
                # Convertir a escala de grises si no lo está ya
                if len(img.shape) == 3:
                    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gris = img.copy()
                
                # Encontrar contornos
                contours, hierarchy = cv2.findContours(gris.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    print(f"No se encontraron contornos en la imagen: {ruta_imagen}")
                    continue
                
                # Seleccionar el contorno con mayor área
                contorno_principal = max(contours, key=cv2.contourArea)
                
                # Calcular área y perímetro
                area = cv2.contourArea(contorno_principal)
                perimetro = cv2.arcLength(contorno_principal, True)
                redondez = (4 * np.pi * area) / (perimetro * perimetro) if perimetro > 0 else 0
                
                # Calcular alargamiento usando cv2.fitEllipse (si el contorno tiene al menos 5 puntos)
                if len(contorno_principal) >= 5:
                    ellipse = cv2.fitEllipse(contorno_principal)
                    eje1, eje2 = ellipse[1]  # ellipse[1] devuelve una tupla (dim1, dim2)
                    eje_mayor = max(eje1, eje2)
                    eje_menor = min(eje1, eje2)
                    alargamiento = eje_mayor / eje_menor if eje_menor > 0 else 0
                else:
                    alargamiento = 0
                
                # Guardar las características para esta imagen
                caracteristicas_forma[clase].append((nombre, redondez, alargamiento))
        
        return caracteristicas_forma

    # Puedes agregar un método para mostrar los resultados en consola o graficar si lo deseas
    def mostrar_caracteristicas_forma(self, caracteristicas_forma):
        """
        Muestra las características extraídas (redondez y alargamiento) para cada imagen.
        """
        for clase, lista in caracteristicas_forma.items():
            print(f"Clase: {clase}")
            for (nombre, redondez, alargamiento) in lista:
                print(f"  Imagen: {nombre} --> Redondez: {redondez:.2f}, Alargamiento: {alargamiento:.2f}")
            print("\n")

    def entrenar_y_evaluar(self):
        """
        Extrae características de color, las normaliza y entrena KMeans.
        En este modo, se usan las características de color (promedio RGB) y se normalizan.
        """
        print("Extrayendo características de color...")
        caracteristicas_color, etiquetas_color = self.extraer_caracteristicas_color(self.segmented_folder)
        print("Aplicando normalización a las características de color...")
        scaler_color = StandardScaler().fit(caracteristicas_color)
        caracteristicas_color_norm = scaler_color.transform(caracteristicas_color)
        dump(scaler_color, "scaler_color.pkl")
        print("Scaler de color guardado: scaler_color.pkl")
        print("Entrenando modelo KMeansManual con características de color normalizadas...")
        kmeans = KMeansManual(n_clusters=self.k, max_iter=100, tol=1e-4)
        kmeans.fit(caracteristicas_color_norm)
        kmeans_labels = kmeans.predict(caracteristicas_color_norm)
        # Asignar etiquetas a clusters mediante voto mayoritario
        etiquetas_clusters = {}
        for i in range(self.k):
            indices_cluster = np.where(kmeans_labels == i)
            etiquetas_reales = etiquetas_color[indices_cluster]
            etiqueta_mayoritaria = max(set(etiquetas_reales), key=list(etiquetas_reales).count)
            etiquetas_clusters[i] = etiqueta_mayoritaria
        dump(kmeans, "kmeans_model.pkl")
        dump(etiquetas_clusters, "kmeans_labels.pkl")
        print("Modelo guardado: kmeans_model.pkl y etiquetas guardadas: kmeans_labels.pkl")
        print("Entrenamiento finalizado.")

    def predecir_imagen_nueva(self, temp_folder):
        """
        Carga una imagen nueva, la preprocesa, segmenta y predice la verdura
        utilizando el modelo KMeans guardado. Se normalizan las características de color.
        Se muestra cada imagen inmediatamente después de la predicción.
        """
        if not os.listdir(temp_folder):
            print(f"Error: La carpeta '{temp_folder}' está vacía. Agrega una imagen para evaluar.")
            return

        carpeta_etiquetada = "ImagenesEtiquetadas"
        os.makedirs(carpeta_etiquetada, exist_ok=True)

        print("Cargando modelo KMeans y scaler de color...")
        try:
            kmeans = load("kmeans_model.pkl")
            etiquetas_clusters = load("kmeans_labels.pkl")
            scaler_color = load("scaler_color.pkl")
        except FileNotFoundError as e:
            print(f"Error: {e}. Asegúrate de haber entrenado y guardado el modelo previamente.")
            return

        procesador = ImageProcessor()

        # Iteramos sobre cada imagen en el folder de nuevas imágenes
        for imagen_nombre in os.listdir(temp_folder):
            ruta_imagen = os.path.join(temp_folder, imagen_nombre)
            imagen = cv2.imread(ruta_imagen)
            if imagen is not None:
                # Preprocesar la imagen (aplicar transformaciones)
                imagen_procesada = procesador.aplicar_transformaciones(imagen)
                # Aplicar segmentación (usa k_segmentation definido en el constructor)
                imagen_segmentada = self.aplicar_kmeans(imagen_procesada)
                # Extraer la característica de color: promedio RGB
                promedio_color = np.mean(imagen_segmentada, axis=(0, 1))
                # Convertir a arreglo de NumPy con forma (1, 3) y normalizar
                caracteristicas_nueva = np.array([promedio_color])
                caracteristicas_nueva_norm = scaler_color.transform(caracteristicas_nueva)
                # Predecir el cluster y obtener la etiqueta correspondiente
                cluster = kmeans.predict(caracteristicas_nueva_norm)[0]
                prediccion = etiquetas_clusters.get(cluster, "Desconocido")
                print(f"Predicción: {prediccion}")
                # Guardar la imagen con la etiqueta predicha
                nombre_etiquetado = f"{prediccion}.jpg"
                ruta_guardado = os.path.join(carpeta_etiquetada, nombre_etiquetado)
                cv2.imwrite(ruta_guardado, imagen)
                
                # Mostrar la imagen inmediatamente (una por una)
                imagen_a_mostrar = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                plt.imshow(imagen_a_mostrar)
                plt.title(f"Predicción: {prediccion}")
                plt.axis("off")
                plt.show()

        # Si deseas quitar la visualización final en cuadrícula, comenta la siguiente línea:
        # self.mostrar_imagenes_etiquetadas(carpeta_etiquetada)


    def mostrar_imagenes_etiquetadas(self, carpeta_etiquetada):
        """
        Muestra todas las imágenes en la carpeta ImagenesEtiquetadas en una cuadrícula.
        """
        import matplotlib.pyplot as plt
        import cv2
        imagenes = [img for img in os.listdir(carpeta_etiquetada) if img.endswith((".jpg", ".png"))]
        if not imagenes:
            print("No se encontraron imágenes etiquetadas.")
            return
        plt.figure(figsize=(12, 8))
        for i, imagen_nombre in enumerate(imagenes):
            ruta_imagen = os.path.join(carpeta_etiquetada, imagen_nombre)
            imagen = cv2.imread(ruta_imagen)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 2, i + 1)
            plt.imshow(imagen)
            plt.title(imagen_nombre.split(".")[0])
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    def evaluar_precision_alternativo(self, folder=None):
        """
        Evalúa la precisión del modelo utilizando el enfoque alternativo:
        para cada cluster, se determina la etiqueta mayoritaria (según las etiquetas reales)
        y se suma el número de imágenes que coinciden con dicha etiqueta.
        
        La precisión se calcula como:
            Precisión = (Suma de imágenes correctas en cada cluster) / (Total de imágenes)
        """
        if folder is None:
            folder = self.segmented_folder
        try:
            kmeans = load("kmeans_model.pkl")
            scaler_color = load("scaler_color.pkl")
        except FileNotFoundError as e:
            print(f"Error: {e}. Asegúrate de haber entrenado y guardado el modelo previamente.")
            return
        print("Extrayendo características de color para evaluación (enfoque alternativo)...")
        caracteristicas, etiquetas_reales = self.extraer_caracteristicas_color(folder)
        caracteristicas_norm = scaler_color.transform(caracteristicas)
        predicted_clusters = kmeans.predict(caracteristicas_norm)
        total_imagenes = len(etiquetas_reales)
        correctos = 0
        cluster_mapping = {}  # Para almacenar la etiqueta mayoritaria de cada cluster
        clusters_unicos = np.unique(predicted_clusters)
        for cluster in clusters_unicos:
            indices = np.where(predicted_clusters == cluster)[0]
            etiquetas_cluster = etiquetas_reales[indices]
            etiqueta_mayoritaria = max(set(etiquetas_cluster), key=lambda x: list(etiquetas_cluster).count(x))
            cluster_mapping[cluster] = etiqueta_mayoritaria
            correctos_cluster = sum(1 for label in etiquetas_cluster if label == etiqueta_mayoritaria)
            correctos += correctos_cluster
        accuracy = correctos / total_imagenes
        print(f"\nPrecisión (enfoque alternativo): {accuracy * 100:.2f}%")
        
        etiquetas_predichas = [cluster_mapping.get(cluster, "Desconocido") for cluster in predicted_clusters]
        cm = confusion_matrix(etiquetas_reales, etiquetas_predichas, labels=np.unique(etiquetas_reales))
        print("\nMatriz de confusión:")
        print(cm)
        print("\nReporte de clasificación:")
        print(classification_report(etiquetas_reales, etiquetas_predichas, labels=np.unique(etiquetas_reales)))

# Ejemplo de uso
if __name__ == "__main__":
    # Se establece k=4 para las 4 clases de verdura.
    # Ahora, por ejemplo, si deseas segmentar con k_segmentation=5, puedes especificarlo:
    procesador = ImageProcessorKMeans(image_folder="ImagenesProcesadas", segmented_folder="ImagenesSegmentadas", k=4, k_segmentation=4)
    caracteristicas_forma = procesador.extraer_caracteristicas_forma(input_folder="ImagenesContorno")
    
    # Mostrar las características extraídas
    procesador.mostrar_caracteristicas_forma(caracteristicas_forma)
    # Procesa y guarda las imágenes segmentadas (usando k_segmentation para segmentación)
    #procesador.procesar_y_guardar_segmentadas()
    
    # Entrena el modelo usando características de color con normalización
    #procesador.entrenar_y_evaluar()
    
    # Evalúa la precisión usando el enfoque alternativo (voto mayoritario por cluster)
    #procesador.evaluar_precision_alternativo()
    
    # Predicción de nuevas imágenes en la carpeta "TempImagenes"
    #procesador.predecir_imagen_nueva(temp_folder="TempImagenes")
