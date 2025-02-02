from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class AudioClassifierKNN:
    def __init__(self, feature_matrix, labels):
        self.feature_matrix = feature_matrix
        self.labels = labels
        # Inicializamos el modelo KNN con 3 vecinos (puedes ajustar este número)
        self.knn = KNeighborsClassifier(n_neighbors=9)

    def entrenar_modelo(self):
        # Dividimos los datos en un conjunto de entrenamiento (80%) y prueba (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            self.feature_matrix, self.labels, test_size=0.2, random_state=42
        )
        
        # Entrenamos el modelo con los datos de entrenamiento
        self.knn.fit(X_train, y_train)
        
        # Realizamos predicciones en el conjunto de prueba
        y_pred = self.knn.predict(X_test)

        # Mostramos el reporte de clasificación
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred))

        # Mostramos la matriz de confusión
        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

        # Mostramos la precisión general del modelo
        print(f"\nExactitud del modelo: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    def predecir(self, nueva_muestra):
        """Método para predecir la categoría de una nueva muestra"""
        return self.knn.predict([nueva_muestra])[0]
