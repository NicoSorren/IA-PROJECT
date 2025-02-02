import numpy as np
from collections import Counter
import pickle

class KnnAlgorithm:
    def __init__(self, k=9):
        """
        Inicializa el algoritmo KNN con el número de vecinos (k).
        """
        self.k = k
        self.features = None
        self.labels = None
        self.model = None  # Añadimos el atributo `model` que contiene el modelo KNN entrenado.

    def fit(self, features, labels):
        """
        Almacena los datos de entrenamiento: características y etiquetas.
        Entrena el modelo KNN con los datos proporcionados.
        """
        self.features = features
        self.labels = labels

        # Entrenamos el modelo KNN utilizando el número de vecinos `k`
        self.model = self.train_knn_model(features, labels)

    def train_knn_model(self, features, labels):
        """
        Entrena el modelo KNN desde cero, sin usar librerías externas como scikit-learn.
        """
        # El modelo KNN usa el número de vecinos `k` y las características de los datos de entrenamiento
        return {"features": features, "labels": labels, "k": self.k}

    def euclidean_distance(self, point1, point2):
        """
        Calcula la distancia euclidiana entre dos puntos (vectores).
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def predict(self, new_sample):
        """
        Realiza una predicción para una nueva muestra utilizando el algoritmo KNN.
        """
        if self.features is None or self.labels is None:
            raise ValueError("El modelo no ha sido entrenado.")
        
        # Calcular la distancia euclidiana entre la muestra nueva y todos los puntos de entrenamiento
        distances = [self.euclidean_distance(new_sample, feature) for feature in self.features]

        # Obtener los índices de los k vecinos más cercanos
        k_indices = np.argsort(distances)[:self.k]

        # Etiquetas de los k vecinos más cercanos
        k_nearest_labels = [self.labels[i] for i in k_indices]

        # Contar la frecuencia de cada etiqueta en los k vecinos más cercanos y devolver la más común
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def evaluate(self, features, labels, file_names=None):
        """
        Evalúa el modelo utilizando un conjunto de prueba.
        Si se proporcionan nombres de archivo, identifica los archivos mal clasificados.
        """
        y_pred = [self.predict(sample) for sample in features]
        
        # Mostrar reporte de clasificación
        print("\nReporte de clasificación:")
        print(self.classification_report(labels, y_pred))

        # Mostrar matriz de confusión
        print("\nMatriz de confusión:")
        print(self.confusion_matrix(labels, y_pred))
        
        # Mostrar exactitud
        accuracy = np.sum(np.array(y_pred) == np.array(labels)) / len(labels)
        print(f"\nExactitud del modelo: {accuracy * 100:.2f}%")
        
        # Identificar errores de clasificación
        incorrectos = [
            {"archivo": file_names[i] if file_names else f"Sample {i}",
            "real": labels[i], "predicho": y_pred[i]}
            for i in range(len(labels)) if labels[i] != y_pred[i]
        ]
        
        if incorrectos:
            print("\nArchivos mal clasificados:")
            for error in incorrectos:
                print(f"Archivo: {error['archivo']}, Real: {error['real']}, Predicho: {error['predicho']}")
        else:
            print("\n¡Todas las muestras fueron clasificadas correctamente!")

        return incorrectos  # Devolver la lista de errores para análisis adicional


    def classification_report(self, y_true, y_pred):
        """
        Genera un reporte de clasificación con precision, recall, f1-score.
        """
        classes = set(y_true)
        report = {}
        for cls in classes:
            tp = sum([1 for i in range(len(y_true)) if y_true[i] == cls and y_pred[i] == cls])
            fp = sum([1 for i in range(len(y_true)) if y_true[i] != cls and y_pred[i] == cls])
            fn = sum([1 for i in range(len(y_true)) if y_true[i] == cls and y_pred[i] != cls])
            tn = sum([1 for i in range(len(y_true)) if y_true[i] != cls and y_pred[i] != cls])

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            
            report[cls] = {'precision': precision, 'recall': recall, 'f1-score': f1_score}
        
        # Generar un string con el reporte
        report_str = ""
        for cls, metrics in report.items():
            report_str += f"{cls} -> Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1-score: {metrics['f1-score']:.2f}\n"
        return report_str

    def confusion_matrix(self, y_true, y_pred):
        """
        Genera una matriz de confusión.
        """
        classes = set(y_true)
        matrix = {cls: {inner_cls: 0 for inner_cls in classes} for cls in classes}
        for i in range(len(y_true)):
            matrix[y_true[i]][y_pred[i]] += 1
        return matrix

    def save_model(self, filename="knn_model.pkl"):
        """Guardar el modelo entrenado en un archivo."""
        with open(filename, "wb") as file:
            pickle.dump(self.model, file)
        print(f"Modelo guardado como {filename}")

    @staticmethod
    def load_model(filename="knn_model.pkl"):
        """
        Cargar un modelo KNN previamente entrenado desde un archivo.
        """
        with open(filename, "rb") as model_file:
            model_data = pickle.load(model_file)
        
        # Crear una nueva instancia de KnnAlgorithm
        knn_model = KnnAlgorithm(k=model_data["k"])  # Usamos el valor de k almacenado
        knn_model.model = model_data  # Cargamos el modelo en el atributo `model`
        
        # Recuperamos las características y las etiquetas y las asignamos a la instancia
        knn_model.features = model_data["features"]
        knn_model.labels = model_data["labels"]
        
        print(f"Modelo cargado desde {filename}")
        return knn_model
    
    

