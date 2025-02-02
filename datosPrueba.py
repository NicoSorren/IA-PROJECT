import numpy as np
import matplotlib.pyplot as plt
from KMeansAlgorithm import KMeansManual  # Importar la clase recién creada

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from KMeansAlgorithm import KMeansManual  # Importa tu clase implementada

# Generar datos sintéticos simples (dos clusters bien definidos)
def generar_datos_sinteticos():
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[8, 8], scale=0.5, size=(50, 2))
    datos = np.vstack((cluster1, cluster2))
    return datos

# Probar KMeansManual
def probar_kmeans_manual():
    # Generar datos
    datos = generar_datos_sinteticos()
    
    # Crear instancia de KMeansManual
    kmeans_manual = KMeansManual(n_clusters=2, max_iter=100)

    
    # Ajustar el modelo a los datos
    centroides, etiquetas = kmeans_manual.fit(datos)

    etiquetas = np.array(etiquetas)

    print(f"Formato de etiquetas: {etiquetas.shape}, Tipo: {type(etiquetas)}")

    # Visualizar resultados
    plt.figure(figsize=(8, 6))
    plt.scatter(datos[:, 0], datos[:, 1], c=etiquetas, cmap='viridis', label='Datos')
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', s=200, marker='X', label='Centroides')
    plt.title("Prueba de KMeansManual")
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.legend()
    plt.show()


def probar_kmeans_manual_datos_adicionales():
    # Datos de prueba con 3 clusters
    X = np.vstack([
        np.random.randn(100, 2) + np.array([2, 2]),
        np.random.randn(100, 2) + np.array([8, 8]),
        np.random.randn(100, 2) + np.array([-5, -5]),
    ])

    # Inicializar y ajustar KMeansManual
    kmeans_manual = KMeansManual(n_clusters=3, max_iter=100)
    centroides, etiquetas = kmeans_manual.fit(X)

    # Validar que etiquetas es un array unidimensional
    etiquetas = np.array(etiquetas).flatten()

    # Graficar datos y centroides
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=etiquetas, cmap='viridis', s=50, label='Datos')
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', s=200, marker='X', label='Centroides')
    plt.title("Prueba de KMeansManual")
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.legend()
    plt.show()


# Ejecutar la prueba
if __name__ == "__main__":
    probar_kmeans_manual_datos_adicionales()
