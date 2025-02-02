import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageProcessor:
    def __init__(self, image_folder="ImagenesVerduras", processed_folder="ImagenesProcesadas"):
        self.image_folder = image_folder
        self.processed_folder = processed_folder
        os.makedirs(self.processed_folder, exist_ok=True)  # Crear la carpeta si no existe

    def cargar_imagenes(self):
        """
        Carga las imágenes desde las carpetas organizadas por verduras.
        """
        imagenes = {}
        for verdura in os.listdir(self.image_folder):
            ruta_verdura = os.path.join(self.image_folder, verdura)
            if os.path.isdir(ruta_verdura):
                imagenes[verdura] = []
                for imagen_nombre in os.listdir(ruta_verdura):
                    ruta_imagen = os.path.join(ruta_verdura, imagen_nombre)
                    imagen = cv2.imread(ruta_imagen)  # Cargar la imagen
                    if imagen is not None:
                        imagenes[verdura].append((imagen_nombre, imagen))  # Guardar nombre e imagen
                    else:
                        print(f"No se pudo cargar la imagen: {ruta_imagen}")
                        
        return imagenes

    def aplicar_transformaciones(self, imagen):
        """
        Aplica transformaciones a la imagen: exposición, contraste, saturación, y nitidez.
        """
        # Aumentar el contraste
        alpha = 1.3 # Factor de contraste (>1 aumenta el contraste)
        beta = 40    # Valor de brillo
        imagen = cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)

        # Aumentar la saturación
        hsv = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 50)  # Aumentar el canal de saturación
        imagen = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Redimensionar la imagen a 224x224
        imagen = cv2.resize(imagen, (224, 224), interpolation=cv2.INTER_AREA)

        # Filtro de nitidez
        kernel = np.array([[0, -1, 0], [-1, 5.3, -1], [0, -1, 0]])
        imagen = cv2.filter2D(imagen, -1, kernel)

        return imagen

    def procesar_y_guardar(self, imagenes):
        """
        Aplica transformaciones y guarda las imágenes en la carpeta ImagenesProcesadas.
        Devuelve un diccionario con imágenes procesadas.
        """
        imagenes_procesadas = {}  # Diccionario para almacenar imágenes procesadas en memoria

        for verdura, lista_imagenes in imagenes.items():
            carpeta_destino = os.path.join(self.processed_folder, verdura)
            os.makedirs(carpeta_destino, exist_ok=True)

            imagenes_procesadas[verdura] = []
            
            for nombre, imagen in lista_imagenes:
                # Aplicar transformaciones
                imagen_procesada = self.aplicar_transformaciones(imagen)
                imagenes_procesadas[verdura].append((nombre, imagen_procesada))  # Guardar en memoria
                
                # Guardar imagen procesada
                ruta_guardado = os.path.join(carpeta_destino, nombre)
                cv2.imwrite(ruta_guardado, cv2.cvtColor(imagen_procesada, cv2.COLOR_RGB2BGR))
                #print(f"Imagen guardada: {ruta_guardado}")
        
        return imagenes_procesadas  # Devuelve imágenes procesadas


    def mostrar_imagenes(self, imagenes, num_por_clase=1):
        """
        Muestra imágenes originales, procesadas y binarizadas en una sola figura.
        Cada fila representa una verdura y muestra las imágenes:
        Original -> Procesada -> Binarizada.
        """
        clases = list(imagenes.keys())  # Lista de nombres de verduras
        total_clases = len(clases)

        plt.figure(figsize=(15, 5 * total_clases))  # Tamaño de la ventana

        for idx, verdura in enumerate(clases):
            #print(f"Mostrando imágenes de: {verdura}")
            carpeta_original = self.image_folder
            carpeta_procesada = self.processed_folder
            carpeta_binarizada = "ImagenesBinarizadas"

            for i, (nombre, _) in enumerate(imagenes[verdura][:num_por_clase]):
                # Rutas de las imágenes
                ruta_original = os.path.join(carpeta_original, verdura, nombre)
                ruta_procesada = os.path.join(carpeta_procesada, verdura, nombre)
                ruta_binarizada = os.path.join(carpeta_binarizada, verdura, f"binarizada_{nombre}")
                
                # Leer imágenes
                img_original = cv2.imread(ruta_original)
                img_procesada = cv2.imread(ruta_procesada)
                img_binarizada = cv2.imread(ruta_binarizada, cv2.IMREAD_GRAYSCALE)
                
                # Convertir imágenes a formato RGB para mostrar correctamente con matplotlib
                img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
                img_procesada = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2RGB)
                
                # Mostrar imágenes
                plt.subplot(total_clases, num_por_clase * 3, idx * num_por_clase * 3 + i * 3 + 1)
                plt.imshow(img_original)
                plt.title(f"{verdura} - Original")
                plt.axis("off")

                plt.subplot(total_clases, num_por_clase * 3, idx * num_por_clase * 3 + i * 3 + 2)
                plt.imshow(img_procesada)
                plt.title(f"{verdura} - Procesada")
                plt.axis("off")

                plt.subplot(total_clases, num_por_clase * 3, idx * num_por_clase * 3 + i * 3 + 3)
                plt.imshow(img_binarizada, cmap="gray")
                plt.title(f"{verdura} - Binarizada")
                plt.axis("off")

        plt.tight_layout()
        plt.show()


    def binarizar_adaptativa(self, imagen):
        """
        Aplica binarización con filtro de ruido, contorno limpio y relleno de agujeros.
        """
        # Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
        
        # Filtro Gaussiano para suavizar ruido
        suavizada = cv2.GaussianBlur(gris, (5, 5), 0)
        
        # Binarización (Otsu)
        _, binarizada = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas
        kernel = np.ones((6, 6), np.uint8)  # Aumentar tamaño del kernel si es necesario
        apertura = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel, iterations=2)  # Eliminar ruido pequeño
        cierre = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel, iterations=3)  # Rellenar agujeros
        
        return cierre
    
    def procesar_y_guardar_binarizadas(self, imagenes, output_folder="ImagenesBinarizadas"):
        """
        Binariza y guarda las imágenes en una carpeta.
        """
        os.makedirs(output_folder, exist_ok=True)  # Crear carpeta si no existe
        
        for verdura, lista_imagenes in imagenes.items():
            carpeta_destino = os.path.join(output_folder, verdura)
            os.makedirs(carpeta_destino, exist_ok=True)
            
            for nombre, imagen in lista_imagenes:
                # Binarizar imagen
                imagen_binarizada = self.binarizar_adaptativa(imagen)
                ruta_guardado = os.path.join(carpeta_destino, f"binarizada_{nombre}")
                cv2.imwrite(ruta_guardado, imagen_binarizada)
                #print(f"Imagen binarizada guardada: {ruta_guardado}")
        return imagen_binarizada
                
# Ejemplo de uso
if __name__ == "__main__":
    procesador = ImageProcessor(image_folder="ImagenesVerduras", processed_folder="ImagenesProcesadas")
    
    # Cargar imágenes originales
    imagenes = procesador.cargar_imagenes()
    
    # Procesar y guardar imágenes (con brillo, saturación, etc.)
    imagenes_procesadas = procesador.procesar_y_guardar(imagenes)
    
    # Aplicar binarización sobre las imágenes procesadas
    procesador.procesar_y_guardar_binarizadas(imagenes_procesadas)

    procesador.mostrar_imagenes(imagenes, num_por_clase=1)