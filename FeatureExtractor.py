import os
import librosa
import numpy as np
import scipy.signal
import logging
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
        self.pca = None  # Inicializamos PCA como None
        self.scaler = StandardScaler()
        self.feature_prueba = []

    def imprimir_estadisticas_por_clase(self, clase, num_componentes=3):
        """
        Imprime estadísticas (media, std, min, max) de las primeras 'num_componentes' componentes
        de la matriz de características (idealmente, la transformada por PCA) para la clase especificada.
        
        Se asume que self.labels es una lista de tuplas en el formato (etiqueta, nombre_archivo)
        y que self.feature_matrix ya contiene las características transformadas (por ejemplo, por PCA)
        si se activó esa opción.
        """
        # Buscar los índices donde la etiqueta (primer elemento de la tupla) coincide con la clase deseada.
        indices = [i for i, lbl in enumerate(self.labels) if lbl[0].lower() == clase.lower()]
        
        if not indices:
            print(f"No se encontraron muestras para la clase '{clase}'.")
            return
        
        # Extraer las filas correspondientes y limitarse a las primeras 'num_componentes' columnas.
        # Si no usaste PCA, estarás viendo las primeras 'num_componentes' características de la matriz escalada.
        datos = self.feature_matrix[indices, :num_componentes]
        
        media = np.mean(datos, axis=0)
        std = np.std(datos, axis=0)
        minimo = np.min(datos, axis=0)
        maximo = np.max(datos, axis=0)
        
        print(f"\nEstadísticas para la clase '{clase}' (primeras {num_componentes} componentes):")
        print(f"Media: {media}")
        print(f"Desviación estándar: {std}")
        print(f"Mínimo: {minimo}")
        print(f"Máximo: {maximo}")

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
        peaks, _ = scipy.signal.find_peaks(magnitude_spectrum, height=20)
        formants = positive_freqs[peaks] * sample_rate
        return formants[:3] if len(formants) >= 3 else [0, 0, 0]

    def extraer_caracteristicas(self, audio, sample_rate, n_mfcc=20, debug=False):
        """
        Extrae características del audio dividiéndolo en 4 segmentos.
        Para cada segmento se calculan:
        - MFCC (n_mfcc coeficientes) y su primer derivada (delta)
        - Spectral Contrast (usando librosa.feature.spectral_contrast)
        Se promedian estas características a lo largo del tiempo y se concatenan.
        Además, se calcula la tasa de cruce por cero (ZCR) y los formantes.
        
        Si debug es True, se pueden registrar detalles para cada segmento.
        """
        num_segmentos = 4
        segmento_duracion = len(audio) // num_segmentos
        segmentos = [audio[i * segmento_duracion:(i + 1) * segmento_duracion] for i in range(num_segmentos)]

        mfcc_features = []
        zcr_features = []
        formants_features = []

        for i, segmento in enumerate(segmentos):
            # Calcular MFCC y delta
            mfcc = librosa.feature.mfcc(
                y=segmento, 
                sr=sample_rate, 
                n_mfcc=n_mfcc,
                n_fft=2048,
                hop_length=512,
            )
            delta_mfcc = librosa.feature.delta(mfcc, width=5, mode='nearest')

            mfcc_mean = np.mean(mfcc, axis=1)
            delta_mean = np.mean(delta_mfcc, axis=1)

            # Calcular Spectral Contrast y promediar a lo largo del tiempo (axis=1)
            spectral_contrast = librosa.feature.spectral_contrast(
                y=segmento, 
                sr=sample_rate, 
                n_fft=1024, 
                hop_length=512
            )
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

            # Concatenar MFCC, delta y spectral contrast para el segmento
            features_segmento = np.concatenate([mfcc_mean, delta_mean, spectral_contrast_mean])
            mfcc_features.append(features_segmento)

            # Calcular ZCR para el segmento
            zcr = self.calcular_zcr(segmento)
            zcr_features.append(zcr)

            # Calcular formantes para el segmento
            formants = self.calcular_formantes(segmento, sample_rate)
            formants_features.extend(formants)

            if debug:
                logging.debug(f"Segmento {i+1}: Primeros 3 MFCC: {mfcc_mean[:3]}")
                logging.debug(f"Segmento {i+1}: Primeros 3 Delta MFCC: {delta_mean[:3]}")
                logging.debug(f"Segmento {i+1}: Spectral Contrast (media): {spectral_contrast_mean}")
                logging.debug(f"Segmento {i+1}: Formantes: {formants}")
                logging.debug(f"Segmento {i+1}: Longitud del segmento: {len(segmento)}")

        # Concatenar las características de todos los segmentos:
        # - mfcc_features: vectores concatenados de MFCC, delta y spectral contrast (por segmento).
        # - zcr_features: un valor de ZCR por segmento.
        # - formants_features: 3 valores de formantes por segmento.
        mfcc_features = np.concatenate(mfcc_features)
        zcr_features = np.array(zcr_features).flatten()
        features = np.concatenate((mfcc_features, zcr_features, np.array(formants_features)))

        logging.debug(f"Estadísticas de características extraídas: min={np.min(features):.3f}, max={np.max(features):.3f}, mean={np.mean(features):.3f}")
        return features

        
    def calcular_pitch(self, audio, sample_rate):
        """
        Estima la frecuencia fundamental (pitch) utilizando librosa.pyin.
        Retorna el valor promedio de F0 a lo largo del audio.
        """
        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'),
                                                        fmax=librosa.note_to_hz('C7'))
            # Elimina los valores NaN (no detectados)
            if f0 is not None:
                f0 = f0[~np.isnan(f0)]
                if len(f0) > 0:
                    return np.mean(f0)
        except Exception as e:
            logging.debug(f"Error al calcular pitch: {e}")
        return 0.0

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

        #pitch_promedio = self.calcular_pitch(audio, sample_rate)
        #logging.debug(f"Pitch promedio: {pitch_promedio}")
        
        try:
            features = [energia] + list(caracteristicas)
            logging.debug(f"Longitud total de features (con energía y pitch): {len(features)}")
            #print(f"[DEBUG] Longitud de características extraídas SIN energía: {len(caracteristicas)}")
            #print(f"[DEBUG] Longitud TOTAL de features (con energía): {len(features)}")
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

    def mostrar_scree_plot(self):
        """
        Si se ha aplicado PCA, muestra un scree plot que ilustra:
          - La varianza explicada por cada componente.
          - La varianza acumulada.
        """
        if self.pca is None:
            print("PCA no se ha aplicado.")
            return

        # Obtener la varianza explicada por cada componente
        evr = self.pca.explained_variance_ratio_
        # Calcular la suma acumulada
        cumulative = np.cumsum(evr)
        
        # Imprimir la varianza explicada y la acumulada
        print("Varianza explicada por cada componente:")
        print(evr)
        print("Varianza acumulada:")
        print(cumulative)
        
        # Crear el scree plot
        plt.figure(figsize=(8, 5))
        componentes = np.arange(1, len(evr) + 1)
        plt.plot(componentes, evr, marker='o', label='Varianza explicada')
        plt.plot(componentes, cumulative, marker='x', label='Varianza acumulada')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Proporción de Varianza')
        plt.title('Scree Plot')
        plt.legend()
        plt.grid(True)
        plt.show()
