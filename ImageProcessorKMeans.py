from KMeansAlgorithm import KMeansManual
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Para el grafico 3D
from ImageProcessor import ImageProcessor
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

class ImageProcessorKMeans:
    def __init__(self, image_folder="ImagenesProcesadas", segmented_folder="ImagenesSegmentadas", k=3):
        self.image_folder = image_folder
        self.segmented_folder = segmented_folder
        self.binarized_folder = "ImagenesBinarizadas"
        self.k = k
        self.kmeans = KMeansManual(n_clusters=self.k, max_iter=100, tol=1e-4)  # Instancia KMeansManual
        self.image_processor = ImageProcessor()  # Instanciar ImageProcessor
        os.makedirs(self.segmented_folder, exist_ok=True)

    def aplicar_kmeans(self, imagen):
        """
        Aplica K-Means Manual para segmentar la imagen en K clusters.
        """
        original_shape = imagen.shape
        imagen_reshape = imagen.reshape((-1, 3))

        # Aplicar K-Means Manual
        self.kmeans.fit(imagen_reshape)  # Ajustar KMeansManual a los datos
        labels = self.kmeans.predict(imagen_reshape)  # Obtener etiquetas para cada punto
        colores_centrales = self.kmeans.centroides  # Centroides como colores representativos

        # Reconstruir la imagen segmentada
        imagen_segmentada = colores_centrales[labels].reshape(original_shape)
        return imagen_segmentada.astype(np.uint8)


    def procesar_y_guardar_segmentadas(self):
        """
        Aplica K-Means a todas las imágenes y las guarda.
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
                        # Aplicar K-Means
                        imagen_segmentada = self.aplicar_kmeans(imagen)

                        # Guardar la imagen segmentada
                        ruta_guardado = os.path.join(carpeta_destino, f"segmentada_{imagen_nombre}")
                        cv2.imwrite(ruta_guardado, cv2.cvtColor(imagen_segmentada, cv2.COLOR_RGB2BGR))
                        #print(f"Imagen segmentada guardada: {ruta_guardado}")

    def extraer_caracteristicas_color(self, folder):
        """
        Extrae las características promedio RGB de las imágenes en una carpeta específica.
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
                        etiquetas.append(verdura)
        return np.array(caracteristicas), np.array(etiquetas)


    def extraer_caracteristicas_forma(self):
        """
        Extrae redondez y alargamiento de las imágenes binarizadas.
        """
        caracteristicas = []  # Lista para almacenar redondez y alargamiento
        etiquetas = []  # Lista para almacenar etiquetas (nombre de la verdura)

        if not os.path.exists(self.binarized_folder):
            print(f"Error: La carpeta '{self.binarized_folder}' no existe. Ejecuta primero el preprocesamiento binarizado.")
            return

        for verdura in os.listdir(self.binarized_folder):
            ruta_verdura = os.path.join(self.binarized_folder, verdura)
            if os.path.isdir(ruta_verdura):
                #print(f"\nCalculando redondez y alargamiento para: {verdura}")
                for imagen_nombre in os.listdir(ruta_verdura):
                    ruta_imagen = os.path.join(ruta_verdura, imagen_nombre)
                    imagen_binarizada = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
                    if imagen_binarizada is not None:
                        # Invertir la imagen si es necesario
                        if np.mean(imagen_binarizada) > 127:  # Si el fondo es oscuro
                            imagen_binarizada = cv2.bitwise_not(imagen_binarizada)

                        # Filtrado adicional para cerrar agujeros
                        kernel = np.ones((5, 5), np.uint8)
                        imagen_binarizada = cv2.morphologyEx(imagen_binarizada, cv2.MORPH_CLOSE, kernel)

                        # Mostrar imagen para verificar
                        #plt.imshow(imagen_binarizada, cmap="gray")
                        #plt.title(f"Imagen binarizada: {imagen_nombre}")
                        #plt.show()

                        # Encontrar contornos externos
                        contornos, _ = cv2.findContours(imagen_binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if len(contornos) > 0:
                            contorno_principal = max(contornos, key=cv2.contourArea)

                            # Calcular redondez
                            area = cv2.contourArea(contorno_principal)
                            perimetro = cv2.arcLength(contorno_principal, True)
                            redondez = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else None

                            # Calcular alargamiento usando la elipse ajustada
                            if len(contorno_principal) >= 5:  # fitEllipse necesita al menos 5 puntos
                                elipse = cv2.fitEllipse(contorno_principal)
                                eje_mayor = max(elipse[1])
                                eje_menor = min(elipse[1])
                                alargamiento = eje_mayor / eje_menor if eje_menor > 0 else None
                            else:
                                alargamiento = None
                            
                                # Guardar características y etiqueta
                            caracteristicas.append([redondez, alargamiento])
                            etiquetas.append(verdura)

                            # Verificar si los valores son válidos
                            #if redondez is not None and alargamiento is not None:
                             #   print(f"{imagen_nombre}: Redondez={redondez:.2f}, Alargamiento={alargamiento:.2f}")
                              #  self.graficar_contornos(imagen_binarizada, contorno_principal, elipse)
                            #else:
                             #   print(f"{imagen_nombre}: No se pudo calcular redondez o alargamiento.")
                        else:
                            print(f"{imagen_nombre}: No se encontraron contornos.")
           # print(f"{np.array(caracteristicas)} ---- {np.array(etiquetas)}")
        return np.array(caracteristicas), np.array(etiquetas)

    def graficar_contornos(self, imagen_binarizada, contorno, elipse):
        """
        Visualiza los contornos y la elipse ajustada en la imagen binarizada.
        """
        imagen_color = cv2.cvtColor(imagen_binarizada, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(imagen_color, [contorno], -1, (0, 255, 0), 2)  # Contorno en verde
        if elipse:
            cv2.ellipse(imagen_color, elipse, (255, 0, 0), 2)  # Elipse en azul

        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(imagen_color, cv2.COLOR_BGR2RGB))
        plt.title("Contorno y Elipse Ajustada")
        plt.axis("off")
        plt.show()



    def entrenar_y_evaluar(self):
        """
        Extrae características de color y forma, las combina, normaliza y entrena KMeans.
        """
        print("Extrayendo características de color...")
        caracteristicas_color, etiquetas_color = self.extraer_caracteristicas_color(self.segmented_folder)

        print("Extrayendo características de forma...")
        caracteristicas_forma, etiquetas_forma = self.extraer_caracteristicas_forma()

        # Verificar si las etiquetas coinciden entre color y forma
        if not np.array_equal(etiquetas_color, etiquetas_forma):
            raise ValueError("Las etiquetas de color y forma no coinciden. Revisa el flujo de extracción.")

        # Normalizar color y forma por separado
        print("Normalizando características...")
        self.scaler_color = StandardScaler().fit(caracteristicas_color)
        self.scaler_forma = StandardScaler().fit(caracteristicas_forma)

        caracteristicas_color_norm = self.scaler_color.transform(caracteristicas_color)
        caracteristicas_forma_norm = self.scaler_forma.transform(caracteristicas_forma)

        # Combinar características normalizadas
        caracteristicas_combinadas = np.hstack((caracteristicas_color_norm, caracteristicas_forma_norm))
        etiquetas = etiquetas_color  # Etiquetas comunes

        # Guardar los scalers y el modelo
        dump(self.scaler_color, "scaler_color.pkl")
        dump(self.scaler_forma, "scaler_forma.pkl")
        print("Scalers guardados: scaler_color.pkl y scaler_forma.pkl")

        # Entrenar KMeansManual
        print("Entrenando modelo KMeansManual...")
        kmeans = KMeansManual(n_clusters=self.k, max_iter=100, tol=1e-4)
        kmeans.fit(caracteristicas_combinadas)

        # Obtener las etiquetas asignadas
        kmeans_labels = kmeans.predict(caracteristicas_combinadas)

        # Asignar etiquetas a clusters
        etiquetas_clusters = {}
        for i in range(self.k):
            indices_cluster = np.where(kmeans_labels == i)
            etiquetas_reales = etiquetas[indices_cluster]
            etiqueta_mayoritaria = max(set(etiquetas_reales), key=list(etiquetas_reales).count)
            etiquetas_clusters[i] = etiqueta_mayoritaria


        # Guardar el modelo y etiquetas
        dump(kmeans, "kmeans_model.pkl")
        dump(etiquetas_clusters, "kmeans_labels.pkl")
        print("Modelo guardado: kmeans_model.pkl y etiquetas guardadas: kmeans_labels.pkl")

        print("Entrenamiento finalizado.")


    def predecir_imagen_nueva(self, temp_folder):
        """
        Carga una imagen nueva, la preprocesa, binariza, segmenta y predice la verdura utilizando un modelo KMeans guardado.
        """
        # Verificar si la carpeta tiene imágenes
        if not os.listdir(temp_folder):
            print(f"Error: La carpeta '{temp_folder}' está vacía. Agrega una imagen para evaluar.")
            return
        
        # Crear carpeta para guardar imágenes etiquetadas
        carpeta_etiquetada = "ImagenesEtiquetadas"
        os.makedirs(carpeta_etiquetada, exist_ok=True)
        # Cargar modelo y scalers guardados
        print("Cargando modelo KMeans y scalers...")
        try:
            kmeans = load("kmeans_model.pkl")
            etiquetas_clusters = load("kmeans_labels.pkl")
            scaler_color = load("scaler_color.pkl")
            scaler_forma = load("scaler_forma.pkl")
        except FileNotFoundError as e:
            print(f"Error: {e}. Asegúrate de haber entrenado y guardado el modelo previamente.")
            return

        procesador = ImageProcessor()

        for imagen_nombre in os.listdir(temp_folder):
            ruta_imagen = os.path.join(temp_folder, imagen_nombre)
            imagen = cv2.imread(ruta_imagen)
            if imagen is not None:
               # print(f"Evaluando imagen: {imagen_nombre}")
                
                # Preprocesar imagen
                imagen_procesada = procesador.aplicar_transformaciones(imagen)
                imagen_binarizada = procesador.binarizar_adaptativa(imagen_procesada)

                if imagen_binarizada is not None:
                    # Invertir la imagen si es necesario
                    if np.mean(imagen_binarizada) > 127:
                        imagen_binarizada = cv2.bitwise_not(imagen_binarizada)

                    kernel = np.ones((5, 5), np.uint8)
                    imagen_binarizada = cv2.morphologyEx(imagen_binarizada, cv2.MORPH_CLOSE, kernel)

                # Extraer características de forma
                contornos, _ = cv2.findContours(imagen_binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contornos) > 0:
                    contorno_principal = max(contornos, key=cv2.contourArea)
                    area = cv2.contourArea(contorno_principal)
                    perimetro = cv2.arcLength(contorno_principal, True)
                    redondez = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else None
                    if len(contorno_principal) >= 5:
                        elipse = cv2.fitEllipse(contorno_principal)
                        eje_mayor = max(elipse[1])
                        eje_menor = min(elipse[1])
                        alargamiento = eje_mayor / eje_menor if eje_menor > 0 else None
                    else:
                        alargamiento = None
                else:
                    print(f"{imagen_nombre}: No se encontraron contornos. Prueba con otra imagen.")
                    return

                # Extraer características de color
                imagen_segmentada = self.aplicar_kmeans(imagen_procesada)
                promedio_color = np.mean(imagen_segmentada, axis=(0, 1))

                # Normalizar características por grupos
                caracteristicas_color_norm = scaler_color.transform([promedio_color])
                caracteristicas_forma_norm = scaler_forma.transform([[redondez, alargamiento]])

                # Combinar características normalizadas
                caracteristicas_nueva = np.hstack([caracteristicas_color_norm, caracteristicas_forma_norm])
                #print(f"Características normalizadas: {caracteristicas_nueva}")

                # Predecir etiqueta
                caracteristicas_nueva = np.array(caracteristicas_nueva).reshape(1, -1)
                cluster = kmeans.predict(caracteristicas_nueva)[0]

                prediccion = etiquetas_clusters.get(cluster, "Desconocido")

                print(f"Predicción: {prediccion}")

                # Guardar imagen etiquetada
                nombre_etiquetado = f"{prediccion}.jpg"
                ruta_guardado = os.path.join(carpeta_etiquetada, nombre_etiquetado)
                cv2.imwrite(ruta_guardado, imagen)
               # print(f"Imagen guardada como {ruta_guardado}")

                # Mostrar las imágenes etiquetadas al final
                #self.mostrar_imagenes_etiquetadas(carpeta_etiquetada)


                #plt.imshow(cv2.cvtColor(imagen_segmentada, cv2.COLOR_BGR2RGB))
                #plt.title(f"Predicción: {prediccion}")
                #plt.axis("off")
                #plt.show()
        self.mostrar_imagenes_etiquetadas(carpeta_etiquetada)

    def mostrar_imagenes_etiquetadas(self, carpeta_etiquetada):
        """
        Muestra todas las imágenes en la carpeta ImagenesEtiquetadas en una cuadrícula.
        """
        import matplotlib.pyplot as plt
        import cv2

        # Obtener todas las imágenes en la carpeta
        imagenes = [img for img in os.listdir(carpeta_etiquetada) if img.endswith((".jpg", ".png"))]

        if not imagenes:
            print("No se encontraron imágenes etiquetadas.")
            return

        # Configurar la figura
        num_imagenes = len(imagenes)
        plt.figure(figsize=(12, 8))  # Tamaño de la figura

        for i, imagen_nombre in enumerate(imagenes):
            ruta_imagen = os.path.join(carpeta_etiquetada, imagen_nombre)
            imagen = cv2.imread(ruta_imagen)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  # Convertir a RGB

            # Mostrar cada imagen en una cuadrícula
            plt.subplot(2, 2, i + 1)  # 2 filas, 2 columnas
            plt.imshow(imagen)
            plt.title(imagen_nombre.split(".")[0])  # Mostrar solo el nombre sin la extensión
            plt.axis("off")  # Ocultar ejes

        plt.tight_layout()
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    procesador = ImageProcessorKMeans(image_folder="ImagenesProcesadas", segmented_folder="ImagenesSegmentadas", k=4)
    procesador.procesar_y_guardar_segmentadas()
    procesador.entrenar_y_evaluar()
    procesador.predecir_imagen_nueva(temp_folder="TempImagenes")