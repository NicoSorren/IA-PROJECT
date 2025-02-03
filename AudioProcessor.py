import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

class AudioProcessor:
    def __init__(self, input_folder="AudiosOriginales", output_folder="AudiosProcesados", silence_threshold=25):
        # Comprobar si la carpeta de salida 'AudiosProcesados' está vacía o no existe
        if os.path.exists("AudiosProcesados") and len(os.listdir("AudiosProcesados")) > 0:
            self.input_folder = "TempAudios"  # Tomar TempAudios si AudiosProcesados no está vacía
        else:
            self.input_folder = input_folder  # De lo contrario, usar AudiosOriginales
        
        self.output_folder = output_folder
        self.silence_threshold = silence_threshold
        os.makedirs(self.output_folder, exist_ok=True)

    def aumentar_volumen(self, audio, ganancia=1):
        """
        Incrementa el volumen del audio multiplicando por un factor de ganancia.
        """
        return audio * ganancia


    def reducir_ruido(self, audio, sample_rate):
        return nr.reduce_noise(
            y=audio, 
            sr=sample_rate, 
            stationary=False, #cambio aca de True a False
            prop_decrease=0.4
        )

    def aplicar_filtro_pasa_alto(self, audio, sample_rate, cutoff=800):
        """
        Aplica un filtro pasa-alto al audio para enfatizar frecuencias altas.
        """
        from scipy.signal import butter, filtfilt
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(1, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, audio)

    def normalizar_audio(self, audio):
        """Normalizar el audio para que tenga una amplitud entre -1 y 1"""
        return audio / np.abs(audio).max() if np.abs(audio).max() != 0 else audio

    def eliminar_silencios(self, archivo_audio):
        """Eliminar silencios y reducir ruido"""
        try:
            ruta_audio = os.path.join(self.input_folder, archivo_audio)
            print(f"ruta_audio es: {ruta_audio}")
        except Exception as e:
            print("error en ruta_audio")
        try:
        # Aseguramos que estamos cargando el archivo correctamente
            audio, sample_rate = librosa.load(ruta_audio, sr=None)
            print(f"Audio cargado correctamente: {archivo_audio}")
        except Exception as e:
            print(f"Error al cargar {archivo_audio}: {e}")
            return
        
        audio_amplificado = self.aumentar_volumen(audio, ganancia=1.5)

        audio_filtrado = self.aplicar_filtro_pasa_alto(audio, sample_rate)
        
        # 1) Normalizar antes de reducir ruido
        audio_normalizado_pre = self.normalizar_audio(audio_filtrado)

        # 2) Reducir ruido sobre el audio ya normalizado
        audio_reducido = self.reducir_ruido(audio_normalizado_pre, sample_rate)

        # Eliminar silencios
        intervalos = librosa.effects.split(audio_reducido, top_db=self.silence_threshold)

        # Añadir margen de 100 ms (ajustable según necesidad)
        margen = int(0.0 * sample_rate)  # 100 ms en muestras. # Con 0.8 me predecia las berenjenas de Pachy

        intervalos_ajustados = []
        for inicio, fin in intervalos:
            inicio_ajustado = max(0, inicio - margen)
            fin_ajustado = min(len(audio_reducido), fin + margen)
            intervalos_ajustados.append((inicio_ajustado, fin_ajustado))
    
    # Unir intervalos ajustados
        audio_sin_silencio = []
        for inicio, fin in intervalos_ajustados:
            audio_sin_silencio.extend(audio_reducido[inicio:fin])

        nombre_salida = f"procesado_{archivo_audio.split('.')[0]}.wav"
        ruta_salida = os.path.join(self.output_folder, nombre_salida)
        sf.write(ruta_salida, audio_sin_silencio, sample_rate)

    def procesar_todos_los_audios(self):
        """Procesar todos los audios en la carpeta de entrada"""
        print(self.input_folder)
        for archivo in os.listdir(self.input_folder):
            if archivo.endswith((".wav", ".ogg")):
                self.eliminar_silencios(archivo)
        
        print("Preprocesamiento completado.")

if __name__ == "__main__":
    procesador = AudioProcessor()
    procesador.procesar_todos_los_audios()
        