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
        Aplica transformaciones a la imagen:
        1. Aísla la verdura del fondo blanco (pone el fondo en negro).
        2. Aumenta la saturación.
        3. Ajusta contraste y brillo.
        4. Redimensiona a 224x224.
        5. Aplica un filtro de nitidez.
        """
        # Paso 1: Aislar la verdura del fondo blanco
        # Definimos un rango para detectar el blanco (puedes ajustar estos valores)
        # Aumentar contraste y brillo
        alpha = 1.2 # Factor de contraste
        beta = 25    # Valor de brillo
        imagen = cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)
        
        lower_white = np.array([140, 140, 140], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        # Crear la máscara: los píxeles que estén en el rango se marcan con 255
        mask = cv2.inRange(imagen, lower_white, upper_white)
        # Invertir la máscara: la verdura (no blanco) tendrá valor 255, el fondo 0
        mask_inv = cv2.bitwise_not(mask)
        # Aplicar la máscara para obtener una imagen donde el fondo sea negro
        imagen_isolada = cv2.bitwise_and(imagen, imagen, mask=mask_inv)

        beta = 40    # Valor de brillo
        imagen_isolada = cv2.convertScaleAbs(imagen_isolada, beta=beta)

        # Paso 2: Aplicar transformaciones a la imagen aislada
        # Convertir la imagen aislada a HSV para aumentar la saturación
        hsv = cv2.cvtColor(imagen_isolada, cv2.COLOR_RGB2HSV)
        # Aumentar la saturación: este valor (140) se puede ajustar según el resultado deseado
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 120)
        # Convertir de vuelta a RGB
        imagen_transformada = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        lab = cv2.cvtColor(imagen_transformada, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab_clahe = cv2.merge((cl, a, b))
        imagen_transformada = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        

        # Redimensionar la imagen a 224x224
        imagen_transformada = cv2.resize(imagen_transformada, (224, 224), interpolation=cv2.INTER_AREA)

        # Aplicar un filtro de nitidez
        kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]])
        imagen_transformada = cv2.filter2D(imagen_transformada, -1, kernel)

        return imagen_transformada


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

            for i, (nombre, _) in enumerate(imagenes[verdura][:num_por_clase]):
                # Rutas de las imágenes
                ruta_original = os.path.join(carpeta_original, verdura, nombre)
                ruta_procesada = os.path.join(carpeta_procesada, verdura, nombre)
                
                # Leer imágenes
                img_original = cv2.imread(ruta_original)
                img_procesada = cv2.imread(ruta_procesada)

                
                # Convertir imágenes a formato RGB para mostrar correctamente con matplotlib
                img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
                img_procesada = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2RGB)
                
                # Mostrar imágenes
                plt.subplot(total_clases, num_por_clase * 2, idx * num_por_clase * 2 + i * 2 + 1)
                plt.imshow(img_original)
                plt.title(f"{verdura} - Original")
                plt.axis("off")

                plt.subplot(total_clases, num_por_clase * 2, idx * num_por_clase * 2 + i * 2 + 2)
                plt.imshow(cv2.cvtColor(img_procesada, cv2.COLOR_BGR2RGB))
                plt.title(f"{verdura} - Procesada")
                plt.axis("off")

                
        plt.tight_layout()
        plt.show()

    
    def binarizar_otsu_con_relleno(self, imagen):
        """
        Convierte la imagen (en BGR) a escala de grises, aplica un desenfoque y
        utiliza Otsu para binarizar la imagen. Luego, usando floodFill, rellena los huecos
        internos sin modificar el contorno del objeto (la verdura). 
        Finalmente, se aplica una operación morfológica de cierre para suavizar los bordes.
        Se muestra la imagen final y se retorna la imagen binaria con los huecos rellenados
        y los bordes suavizados.
        """
        # 1. Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # 2. Suavizar con desenfoque gaussiano
        suavizada = cv2.GaussianBlur(gris, (5, 5), 0)
        
        # 3. Binarización con Otsu
        _, binaria = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Rellenar huecos internos usando floodFill
        floodfill = binaria.copy()
        h, w = binaria.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(floodfill, mask, (0, 0), 255)
        floodfill_inv = cv2.bitwise_not(floodfill)
        binaria_rellena = binaria | floodfill_inv
        
        # 5. Suavizar los bordes aplicando cierre morfológico
        kernel_smooth = np.ones((3, 3), np.uint8)
        binaria_rellena = cv2.morphologyEx(binaria_rellena, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
        
        # 6. Mostrar la imagen resultante
        #plt.figure()
        #plt.imshow(binaria_rellena, cmap="gray")
        #plt.title("Imagen Binarizada con Otsu, Huecos Rellenados y Bordes Suavizados")
        #plt.axis("off")
        #plt.show()
        
        return binaria_rellena
    
    def procesar_y_guardar_binarizadas(self, imagenes, output_folder="ImagenesBinarizadas"):
        """
        Aplica la binarización con Otsu, relleno de huecos y suavizado de bordes a cada imagen,
        y guarda las imágenes resultantes en la carpeta 'ImagenesBinarizadas' manteniendo la estructura
        por clases. Devuelve un diccionario con las imágenes binarizadas.
        """
        imagenes_binarizadas = {}  # Diccionario para almacenar las imágenes binarizadas en memoria

        # Crear la carpeta de salida, si no existe. Si existe, se puede sobreescribir.
        os.makedirs(output_folder, exist_ok=True)

        for verdura, lista_imagenes in imagenes.items():
            carpeta_destino = os.path.join(output_folder, verdura)
            os.makedirs(carpeta_destino, exist_ok=True)

            imagenes_binarizadas[verdura] = []

            for nombre, imagen in lista_imagenes:
                # Aplicar el método de binarización con Otsu, relleno y suavizado de bordes
                imagen_binarizada = self.binarizar_otsu_con_relleno(imagen)
                imagenes_binarizadas[verdura].append((nombre, imagen_binarizada))

                # Guardar la imagen binarizada en la carpeta de salida
                ruta_guardado = os.path.join(carpeta_destino, nombre)
                cv2.imwrite(ruta_guardado, imagen_binarizada)

        return imagenes_binarizadas

    
    def mostrar_y_guardar_imagenes_binarizadas_otsu_relleno(self, num_por_clase=1):
        """
        Recorre todas las imágenes en la carpeta de imágenes procesadas,
        les aplica la binarización con Otsu, relleno de huecos y suavizado de bordes,
        guarda las imágenes resultantes en la carpeta "ImagenesBinarizadas"
        (creándola o limpiándola si ya existe) y muestra todas las imágenes binarizadas en una cuadrícula.
        
        num_por_clase: número de imágenes por clase a procesar y mostrar.
        """
        output_folder = "ImagenesBinarizadas"
        # Crear la carpeta si no existe; si existe, eliminar archivos anteriores
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            for file in os.listdir(output_folder):
                ruta_file = os.path.join(output_folder, file)
                try:
                    os.remove(ruta_file)
                except Exception as e:
                    print(f"No se pudo eliminar {ruta_file}: {e}")

        clases = os.listdir(self.processed_folder)
        total_clases = len(clases)
        
        plt.figure(figsize=(15, 5 * total_clases))
        
        for idx, clase in enumerate(clases):
            carpeta_clase = os.path.join(self.processed_folder, clase)
            imagenes = os.listdir(carpeta_clase)
            
            # Mostrar y guardar hasta num_por_clase imágenes por clase
            for i, nombre in enumerate(imagenes[:num_por_clase]):
                ruta_imagen = os.path.join(carpeta_clase, nombre)
                imagen = cv2.imread(ruta_imagen)  # Se carga en BGR
                if imagen is not None:
                    imagen_binarizada = self.binarizar_otsu_con_relleno(imagen)
                    
                    # Guardar la imagen binarizada en la carpeta "ImagenesBinarizadas"
                    # Puedes formar el nombre como "clase_binarizada_nombre"
                    nombre_guardado = f"{clase}_binarizada_{nombre}"
                    ruta_guardado = os.path.join(output_folder, nombre_guardado)
                    cv2.imwrite(ruta_guardado, imagen_binarizada)
                    
                    # Ubicar la imagen en la cuadrícula para mostrarla
                    plt.subplot(total_clases, num_por_clase, idx * num_por_clase + i + 1)
                    plt.imshow(imagen_binarizada, cmap="gray")
                    plt.title(f"{clase} - {nombre}")
                    plt.axis("off")
                else:
                    print(f"No se pudo leer la imagen: {ruta_imagen}")
        
        plt.tight_layout()
        plt.show()
    
    def detectar_contornos_canny(self, imagen):
        """
        Aplica el detector de bordes Canny a la imagen.
        Si la imagen está en color, se convierte a escala de grises.
        Retorna el mapa de bordes.
        """
        # Si la imagen tiene 3 canales, convertir a escala de grises
        if len(imagen.shape) == 3:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gris = imagen
        # Aplicar el detector de bordes Canny
        edges = cv2.Canny(gris, 50, 150)
        return edges

    
    def procesar_y_guardar_contornos_canny(self, input_folder="ImagenesBinarizadas", output_folder="ImagenesContorno", num_por_clase=1):
        """
        Recorre todas las imágenes en la carpeta 'input_folder' (por defecto, ImagenesBinarizadas),
        aplica el detector de bordes Canny para extraer los contornos, y guarda los mapas de bordes
        resultantes en la carpeta 'output_folder', manteniendo la estructura de carpetas por clase.
        Además, muestra todas las imágenes resultantes en una cuadrícula.
        
        num_por_clase: número de imágenes por clase a procesar y mostrar.
        """
        # Crear la carpeta de salida si no existe; si existe, limpiar su contenido (solo archivos)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    try:
                        os.remove(os.path.join(root, file))
                    except Exception as e:
                        print(f"No se pudo eliminar {file}: {e}")
                        
        # Listar las subcarpetas (clases) en el input_folder
        clases = os.listdir(input_folder)
        total_clases = len(clases)
        
        # Preparar figura para mostrar
        plt.figure(figsize=(15, 5 * total_clases))
        
        for idx, clase in enumerate(clases):
            carpeta_input = os.path.join(input_folder, clase)
            carpeta_output = os.path.join(output_folder, clase)
            os.makedirs(carpeta_output, exist_ok=True)
            
            imagenes = os.listdir(carpeta_input)
            for i, nombre in enumerate(imagenes[:num_por_clase]):
                ruta_imagen = os.path.join(carpeta_input, nombre)
                # Leer la imagen (se asume que está en BGR, ya que se guardó con cv2.imwrite)
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"No se pudo leer la imagen: {ruta_imagen}")
                    continue

                # Aplicar el detector de bordes Canny usando tu método definido
                edges = self.detectar_contornos_canny(imagen)
                
                # Guardar la imagen resultante en output_folder
                ruta_guardado = os.path.join(carpeta_output, nombre)
                cv2.imwrite(ruta_guardado, edges)
                
                # Mostrar la imagen en la cuadrícula
                plt.subplot(total_clases, num_por_clase, idx * num_por_clase + i + 1)
                plt.imshow(edges, cmap="gray")
                plt.title(f"{clase} - {nombre}")
                plt.axis("off")
        
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    procesador = ImageProcessor(image_folder="ImagenesVerduras", processed_folder="ImagenesProcesadas")
    
    # Cargar imágenes originales
    imagenes = procesador.cargar_imagenes()
    
    # Procesar y guardar imágenes (transformaciones)
    imagenes_procesadas = procesador.procesar_y_guardar(imagenes)
    
    # Mostrar imágenes originales y procesadas
    procesador.mostrar_imagenes(imagenes, num_por_clase=1)
    
    # Procesar y guardar las imágenes binarizadas en "ImagenesBinarizadas"
    imagenes_binarizadas = procesador.procesar_y_guardar_binarizadas(imagenes_procesadas)

    procesador.procesar_y_guardar_contornos_canny(num_por_clase=10)
