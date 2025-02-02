import numpy as np

class KMeansManual:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        """
        Clase para implementar el algoritmo KMeans manualmente.
        Parámetros:
            n_clusters: int
                Número de clusters (k).
            max_iter: int
                Número máximo de iteraciones.
            tol: float
                Tolerancia para la convergencia.
            random_state: int
                Semilla para la reproducibilidad.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroides = None
        self.labels = None

    def _inicializar_centroides(self, X):
        """
        Inicializa los centroides aleatoriamente seleccionando k puntos del dataset.
        """
        if self.random_state:
            np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _calcular_distancias(self, X, centroides):
        """
        Calcula la distancia euclidiana entre cada punto y cada centroide.
        """
        return np.linalg.norm(X[:, np.newaxis] - centroides, axis=2)

    def _actualizar_centroides(self, X, labels):
        """
        Actualiza los centroides calculando el promedio de los puntos asignados a cada cluster.
        """
        nuevos_centroides = []
        for i in range(self.n_clusters):
            puntos_cluster = X[labels == i]
            if len(puntos_cluster) > 0:
                nuevos_centroides.append(puntos_cluster.mean(axis=0))
            else:
                nuevos_centroides.append(np.zeros(X.shape[1]))
        return np.array(nuevos_centroides)

    def fit(self, X):
        """
        Entrena el modelo KMeans con los datos de entrada X.
        Parámetros:
            X: np.array
                Datos de entrada (n_samples, n_features).
        """
        # Inicialización de centroides
        self.centroides = self._inicializar_centroides(X)

        for i in range(self.max_iter):
            # Paso 1: Calcular distancias y asignar clusters
            distancias = self._calcular_distancias(X, self.centroides)
            self.labels = np.argmin(distancias, axis=1)

            # Paso 2: Actualizar centroides
            nuevos_centroides = self._actualizar_centroides(X, self.labels)

            # Paso 3: Verificar convergencia
            if np.all(np.linalg.norm(nuevos_centroides - self.centroides, axis=1) < self.tol):
                #print(f"Convergencia alcanzada en {i + 1} iteraciones.")
                break

            self.centroides = nuevos_centroides

        return self.centroides, self.labels

    def predict(self, X):
        """
        Predice el cluster más cercano para cada punto en X.
        Parámetros:
            X: np.array
                Nuevos datos de entrada (n_samples, n_features).
        Retorna:
            np.array: Clusters asignados a cada punto.
        """
        distancias = self._calcular_distancias(X, self.centroides)
        return np.argmin(distancias, axis=1)
