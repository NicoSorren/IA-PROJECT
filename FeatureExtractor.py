import os
import librosa
import numpy as np
import scipy.signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FeatureExtractor:
    def __init__(self, input_folder="AudiosProcesados", use_pca=True, n_components=9):
        self.input_folder = input_folder
        self.feature_matrix = []
        self.labels = []
        self.use_pca = use_pca
        self.n_components = n_components
        # self.feature_length = 15
        self.pca = None  # Inicializamos PCA como None
        self.scaler = StandardScaler()  # Inicializamos el scaler. # StandardScaler estandariza los datos a una distribución con media 0 y desviación estándar 1.
        self.feature_prueba = []

    def calcular_energia(self, audio):
        energia = np.sum(audio**2) / len(audio)
        return np.log(energia + 1e-10)  # Se añade un epsilon para evitar log(0)

    def calcular_formantes(self, audio_segment, sample_rate):
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio_segment[0], audio_segment[1:] - pre_emphasis * audio_segment[:-1])
        hamming_window = np.hamming(len(emphasized_audio))
        fft_spectrum = np.fft.fft(emphasized_audio * hamming_window)
        freqs = np.fft.fftfreq(len(fft_spectrum))
        positive_freqs = freqs[:len(freqs) // 2]
        magnitude_spectrum = np.abs(fft_spectrum[:len(fft_spectrum) // 2])
        peaks, _ = scipy.signal.find_peaks(magnitude_spectrum, height=0)
        formants = positive_freqs[peaks] * sample_rate
        return formants[:2] if len(formants) >= 2 else [0, 0]

    def extraer_caracteristicas(self, audio, sample_rate, n_mfcc=13, debug=False):
        """
        Este método extrae características tanto para los audios de entrenamiento como para el audio de prueba.
        Se elimina la segmentación del primer tercio y se agregan formantes a todos los audios.
        """
        # Dividir el audio en partes iguales (por ejemplo, en 3 partes)
        num_segmentos = 10
        segmento_duracion = len(audio) // num_segmentos
        segmentos = [audio[i*segmento_duracion:(i+1)*segmento_duracion] for i in range(num_segmentos)]

        # Extraer características de cada segmento
        mfcc_features = []
        zcr_features = []
        formants_features = []  # Para guardar los formantes de cada segmento

        for i, segmento in enumerate(segmentos):
            mfcc = librosa.feature.mfcc(
            y=segmento, 
            sr=sample_rate, 
            n_mfcc=13,
            n_fft=2048,    # Reducir de 2048 (valor por defecto)
            hop_length=512 # Reducir de 512
            )
            delta_mfcc = librosa.feature.delta(mfcc, width=5, mode='nearest')             # 1ra derivada
            delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=5, mode='nearest')   # 2da derivada

            # Promediar y concatenar todas las características del segmento
            mfcc_mean = np.mean(mfcc, axis=1)
            delta_mean = np.mean(delta_mfcc, axis=1)
            delta2_mean = np.mean(delta2_mfcc, axis=1)

            # Si debug está activo, mostramos información del segmento
            if debug:
                print(f"[DEBUG] Segmento {i+1}: Primeros 3 MFCC: {mfcc_mean[:3]}")
                print(f"[DEBUG] Segmento {i+1}: Primeros 3 Delta MFCC: {delta_mean[:3]}")
                print(f"[DEBUG] Segmento {i+1}: Primeros 3 Delta2 MFCC: {delta2_mean[:3]}")

            mfcc_features.append(np.concatenate([mfcc_mean, delta_mean, delta2_mean]))  # ¡Aquí está el cambio!

            # Calcular ZCR para cada segmento
            zcr = self.calcular_zcr(segmento)
            zcr_features.append(zcr)
            
            # Calcular formantes para el segmento
            formants = self.calcular_formantes(segmento, sample_rate)
            if debug:
                print(f"[DEBUG] Segmento {i+1}: Formantes: {formants}")
            # Aquí, suponiendo que la función devuelve dos valores,
            # se agregan de forma consecutiva para cada segmento.
            formants_features.extend(formants)
            
        # Aplanar la lista de características MFCC
        mfcc_features = np.concatenate(mfcc_features)
        zcr_features = np.array(zcr_features).flatten()

        # Calcular formantes (siempre se agregan)
        #formantes = self.calcular_formantes(audio, sample_rate)
        
        # Concatenar MFCC, ZCR y formantes
        features = np.concatenate((mfcc_features, zcr_features, np.array(formants_features)))

        print(f"[DEBUG] Estadísticas de características extraídas: min={np.min(features):.3f}, max={np.max(features):.3f}, mean={np.mean(features):.3f}")

        #print(f"Número de características: {len(features)}")
        
        return features

    
    def procesar_todos_los_audios(self):
        """Este método ahora se puede usar tanto para un directorio de audios como para un solo archivo"""
        print(f"Verificando la ruta de entrada: {self.input_folder}")

        if os.path.isdir(self.input_folder):    # verifica si input_folder es un directorio
            # Listar todos los archivos en la carpeta para asegurarse de que están ahí
            archivos_en_directorio = os.listdir(self.input_folder)
            if not archivos_en_directorio:
                print("¡Advertencia! El directorio está vacío.")

            for archivo in os.listdir(self.input_folder):
                if archivo.endswith(".wav"):
                    #print(f"archivo antes de procesar_audio es:{archivo}")
                    self.procesar_audio(archivo)
            
            self.feature_matrix = np.array(self.feature_matrix)     # convierte self.feature_matrix (que hasta este punto es una lista de listas) en un array de NumPy.
            print(f"Características extraídas de {len(self.feature_matrix)} archivos.")

            self.feature_matrix = self.scaler.fit_transform(self.feature_matrix) # fit_transform ajusta el scaler a los datos (calcula media y desviación estándar) y luego aplica la transformación

            if self.use_pca:
                self.pca = PCA(n_components=self.n_components)      # Si self.use_pca es True, aplica el algoritmo PCA para reducir el número de características (dimensiones) de los datos. n_components determina el número de dimensiones retenidas.
                self.feature_matrix = self.pca.fit_transform(self.feature_matrix)   # fit_transform ajusta el PCA a los datos y los transforma, produciendo una nueva matriz donde cada fila representa un audio y cada columna representa un componente principal.
                #print("PCA aplicado. Componentes retenidos:", self.n_components)
        
        else:
            try:
                print(f"input_folder es: {self.input_folder}")
                self.feature_prueba = self.procesar_audio(self.input_folder)
            except Exception as e:
                print("No se pudo ejecutar procesar_audios con audio de prueba")

                                
        return self.feature_matrix, self.labels, self.feature_prueba

    def procesar_audio(self, archivo_audio):
        if os.path.isdir(self.input_folder):
            ruta_audio = os.path.join(self.input_folder, archivo_audio)
        else:
            try:
                ruta_audio = self.input_folder
                print(f"ruta_audio es: {ruta_audio}")
            except Exception as e:
                print("No se ha podido modificar la ruta_audio para audio de prueba")

        try:
            audio, sample_rate = librosa.load(ruta_audio, sr=None)
        except Exception as e:
            print(f"Error al cargar {ruta_audio}: {e}")

        try:
            energia = self.calcular_energia(audio)
        except Exception as e:
            print("No se ha aplicado calcular_energia")

        # Activar depuración si el archivo corresponde a la clase "camote"
        debug_flag = "camote" in archivo_audio.lower()

        # Extraer las características; se pasa debug_flag
        caracteristicas = self.extraer_caracteristicas(audio, sample_rate, n_mfcc=13, debug=debug_flag)
        
        try:
            features = [energia] + list(caracteristicas)
            print(f"[DEBUG] Longitud de características extraídas SIN energía: {len(caracteristicas)}")
            print(f"[DEBUG] Longitud TOTAL de features (con energía): {len(features)}")
        except Exception as e:
            print("NO se ha aplicado la suma de features (energía + características)")

        if os.path.isdir(self.input_folder):
            self.feature_matrix.append(features)
            self.labels.append((archivo_audio.split("_")[1], archivo_audio))
        else:
            try:
                features = np.array(features).reshape(1, -1)
            except Exception as e:
                print("No se ha podido aplicar np.array")

            try:
                if not hasattr(self.scaler, 'mean_'):
                    raise ValueError("El escalador no está ajustado. Asegúrate de procesar los datos de entrenamiento primero.")
                features = self.scaler.transform(features)
            except Exception as e:
                print(f"No se ha podido aplicar transform: {e}")
            
            if self.pca:
                try:
                    features = self.pca.transform(features)
                except Exception as e:
                    print("No se ha podido aplicar PCA")
            print(f"Características extraídas y transformadas del archivo de prueba: {features}")
       
        return features


    def calcular_zcr(self, audio):
        """Calcula la tasa de cruce por cero (ZCR) de un segmento de audio."""
        return np.mean(librosa.feature.zero_crossing_rate(y=audio).flatten())


    def visualizar_caracteristicas_3d_con_etiquetas(self):
        """
        Muestra una gráfica 3D de las características (tras PCA)
        para las cuatro clases de verduras: 'papa', 'berenjena', 'zanahoria' y 'camote'.
        """

        # 1. Extraer solo la clase de self.labels (ignorando el nombre del archivo)
        #    labels_reales será algo como ["papa", "berenjena", "papa", "camote", ...]
        labels_reales = [lbl[0] for lbl in self.labels]

        # 2. Definir las clases y sus colores correspondientes
        #    (Si cambias la clase, asegúrate de añadirla aquí)
        clases = ["papa", "berenjena", "zanahoria", "camote"]
        colores = {
            "papa": "yellow",
            "berenjena": "green",
            "zanahoria": "red",
            "camote": "orange"
        }

        # Crear la figura en 3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # 3. Para cada clase, filtra los índices que le correspondan y grafica esos puntos
        for clase in clases:
            # Obtener índices donde la etiqueta real sea la clase actual
            indices = [i for i, c in enumerate(labels_reales) if c == clase]

            if not indices:
                # Si no hay muestras de esa clase, omitimos la graficación
                continue

            # Extraer las 3 primeras componentes principales (x, y, z) de tu feature_matrix
            x = self.feature_matrix[indices, 0]
            y = self.feature_matrix[indices, 1]
            z = self.feature_matrix[indices, 2]

            # Graficar puntos de esta clase con su color y etiqueta
            ax.scatter(
                x, y, z,
                c=colores[clase], 
                label=clase,
                alpha=0.7
            )

        # Configurar ejes y título
        ax.set_title("Características de Audio - PCA 3D")
        ax.set_xlabel("Componente principal 1")
        ax.set_ylabel("Componente principal 2")
        ax.set_zlabel("Componente principal 3")

        # Mostrar leyenda con el nombre de cada clase
        ax.legend()

        plt.show()
