import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

class LiveAudioProcessor:
    def __init__(self, filename="TempAudios/audio.wav", duration=3, sample_rate=44100):
        self.filename = filename
        self.duration = duration
        self.sample_rate = sample_rate

    def grabar_audio(self):
        """
        Graba el audio en vivo durante el tiempo especificado y guarda el archivo en WAV.
        """
        print(f"Grabando audio en vivo durante {self.duration} segundos...")
        
        # Graba el audio en vivo
        audio = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()  # Esperar hasta que termine la grabación

        # Guarda el archivo en la carpeta TempAudios como audio.wav
        wav.write(self.filename, self.sample_rate, (audio * 32767).astype(np.int16))  # Convertir a formato WAV
        print(f"Audio grabado y guardado como {self.filename}")

    def eliminar_audio(self):
        """
        Elimina el archivo de audio después de la ejecución.
        """
        if os.path.exists(self.filename):
            os.remove(self.filename)
            print(f"Archivo {self.filename} eliminado.")

