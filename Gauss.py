import numpy as np


def gaussian_elimination(A, b):
    """
    Resuelve el sistema de ecuaciones Ax = b utilizando eliminación gaussiana.

    Pasos del algoritmo:
    1. Pivoteo parcial: Intercambia filas para reducir errores numéricos.
    2. Normalización del pivote: Convierte el pivote en 1.
    3. Eliminación hacia adelante: Crea una matriz triangular superior.
    4. Sustitución hacia atrás: Resuelve el sistema triangular.

    :param A: Matriz de coeficientes (numpy array de tamaño n x n)
    :param b: Vector de términos independientes (numpy array de tamaño n)
    :return: Vector solución x (numpy array de tamaño n)
    """
    n = len(A)
    A = A.astype(float)  # Convertimos la matriz a tipo float para evitar errores en divisiones
    b = b.astype(float)  # Convertimos el vector b a tipo float

    # Eliminación hacia adelante
    for i in range(n):
        # Pivoteo parcial: buscamos el mayor valor absoluto en la columna i para reducir errores numéricos
        max_row = np.argmax(abs(A[i:, i])) + i  # Encontramos el índice del máximo valor en la columna i
        A[[i, max_row]] = A[[max_row, i]]  # Intercambiamos las filas
        b[[i, max_row]] = b[[max_row, i]]  # Intercambiamos los elementos de b

        # Hacemos que A[i, i] sea 1 dividiendo la fila por A[i, i]
        pivot = A[i, i]
        if pivot == 0:
            raise ValueError("El sistema no tiene solución única")
        A[i] = A[i] / pivot
        b[i] = b[i] / pivot

        # Eliminamos los elementos debajo del pivote
        for j in range(i + 1, n):
            factor = A[j, i]  # Factor por el cual se multiplicará la fila pivote
            A[j] = A[j] - factor * A[i]  # Restamos la fila escalada para eliminar el coeficiente
            b[j] = b[j] - factor * b[i]  # Ajustamos el vector b en consecuencia

    # Sustitución hacia atrás
    x = np.zeros(n)  # Inicializamos el vector solución
    for i in range(n - 1, -1, -1):  # Iteramos desde la última fila hacia atrás
        x[i] = b[i] - np.dot(A[i, i + 1:], x[i + 1:])  # Calculamos el valor de x[i]

    return x


# Ejemplo de uso
A = np.array([[2, -1, 1], [3, 2, -4], [1, 1, 1]])  # Matriz de coeficientes
b = np.array([1, 2, 3])  # Vector de términos independientes
x = gaussian_elimination(A, b)  # Llamamos a la función
print("Solución:", x)  # Imprimimos la solución
