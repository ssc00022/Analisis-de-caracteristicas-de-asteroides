"""
Nombre del proyecto: Análisis predictivo y descriptivo de características de asteroides y sus trayectorias
Archivo: analisis.py
Autor: Samuel Sánchez Carrasco
Fecha: 25/08/2024
Descripción: Funciones generales de carga y división de datos, cálculo de métricas y visualización de resultados
"""

import numpy as np
import pandas as pd
import seaborn as sns
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, mean_squared_error,
                             root_mean_squared_error, mean_absolute_error, r2_score, silhouette_score,
                             cohen_kappa_score, ConfusionMatrixDisplay)
from matplotlib import pyplot as plt
from pandas.plotting import table
import plotly.express as px
import plotly.graph_objects as go


def load_dataset(filename):
    """
    Carga un archivo CSV en un DataFrame de pandas.

    Esta función carga un archivo dado por el nombre 'filename' en un DataFrame de pandas.
    Luego, visualiza la información general del DataFrame y muestra las primeras filas.

    Args:
        filename (str): Nombre del archivo que contiene el conjunto de datos.

    Returns:
        pd.DataFrame: DataFrame cargado con los datos del archivo.
    """
    # Carga del dataset
    df = pd.read_csv('datasets/' + filename + '.csv')

    # Información general del dataframe
    df.info()

    # Primeras filas del dataframe
    print(df.head())

    return df


def prediction_metrics(metrics, y_test, y_pred, classification, algorithm):
    """
    Actualiza un diccionario de métricas según el algoritmo aplicado.

    Si el algoritmo es de clasificación, se calculan las métricas de accuracy, precision, recall y f1-score. Si el
    algoritmo es de regresión, se calculan las métricas MSE, RMSE, MAE y R² Score individualmente. El diccionario se
    actualiza con un nuevo valor en cada medida para más tarde calcular el promedio de cada una de ellas.

    Args:
        metrics (dict): Diccionario con algoritmos y métricas
        y_test (list): Lista de etiquetas del conjunto de test
        y_pred (list): Lista de etiquetas predecidas por el algoritmo
        classification (bool): Booleano que indica si las métricas son de clasificación (True) o regresión (False)
        algorithm (str): Nombre del algoritmo aplicado

    Returns:
        dict: Diccionario de métricas actualizadas
    """
    if classification:
        # Métricas de clasificación
        metrics[algorithm]['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics[algorithm]['Precision'].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        metrics[algorithm]['Recall'].append(recall_score(y_test, y_pred, average='macro'))
        metrics[algorithm]['F1-Score'].append(f1_score(y_test, y_pred, average='macro'))
    else:
        # Métricas de regresión
        metrics[algorithm]['MSE'].append(mean_squared_error(y_test, y_pred))
        metrics[algorithm]['RMSE'].append(root_mean_squared_error(y_test, y_pred))
        metrics[algorithm]['MAE'].append(mean_absolute_error(y_test, y_pred))
        metrics[algorithm]['R² Score'].append(r2_score(y_test, y_pred))

    return metrics


def show_tables(data_list):
    """
    Recibe una lista de diccionarios de datasets y los muestra en imágenes de tablas.

    Args:
        data_list (list): Lista de diccionarios cuya clave es el nombre del DataFrame y su valor un pd.DataFrame
    """
    for item in data_list:
        for title, df in item.items():  # Iterar sobre los elementos del diccionario
            # Calcular el tamaño de la figura basado en el tamaño del DataFrame
            num_rows, num_cols = df.shape
            fig, ax = plt.subplots(figsize=(15, 15), num=title)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_frame_on(False)
            ax.set_title(title, pad=10, y=0.7)

            # Crear la tabla en los ejes
            tab = table(ax, df, loc='center', colWidths=[0.2] * num_cols)
            tab.auto_set_font_size(False)
            tab.set_fontsize(10)
            tab.scale(2, 2)  # Ajustar el tamaño de la tabla

    plt.show()  # Mostrar todas las tablas en ventanas separadas
    plt.close('all')  # Cerrar todas las ventanas abiertas


def kappa_heatmap(y_predictions):
    """
    Recibe un diccionario con las etiquetas predecidas de cada algoritmo y calcula el coeficiente kappa de Cohen para
    cada par de algritmos, completando una matriz con dichos valores que, tras ser transformada en un pd.DataFrame,
    servirá de entrada para generar un mapa de calor.

    Args:
        y_predictions (dict): Diccionario cuyas claves representan algoritmos y cuyos valores contienen etiquetas predecidas
    """
    # Lista de algoritmos
    algorithms = list(y_predictions.keys())

    # Crear matriz para almacenar puntuaciones kappa
    kappa_matrix = np.zeros((len(algorithms), len(algorithms)))

    # Rellenar matriz con puntuaciones kappa
    for i, algo1 in enumerate(algorithms):
        for j, algo2 in enumerate(algorithms):
            if i != j:  # Calcular puntuación kappa para algoritmos diferentes
                kappa_matrix[i, j] = cohen_kappa_score(y_predictions[algo1], y_predictions[algo2])
            else:
                kappa_matrix[i, j] = 1.0  # Puntuación kappa para algoritmo consigo mismo

    # Mostrar la matriz de kappa como un mapa de calor
    kappa_df = pd.DataFrame(kappa_matrix, index=algorithms, columns=algorithms)
    plt.figure(figsize=(8, 6))
    sns.heatmap(kappa_df, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Puntuaciones Kappa entre clasificadores')
    plt.show()


def plot_confusion_matrix(confusion_matrix, class_labels):
    """
    Recibe un diccionario con las matrices de confusión de cada algoritmo y los genera individualmente.

    Args:
        confusion_matrix (dict): Diccionario cuyas claves representan algoritmos y cuyos valores contienen matrices de confusión
        class_labels (list): Lista para mostrar las clases en las matrices de confusión
    """
    # Iterar sobre cada clasificador y generar sus matrices de confusión
    for classifier in confusion_matrix.keys():
        disp = ConfusionMatrixDisplay(confusion_matrix[classifier], display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues, colorbar=False)
        plt.xlabel('')
        plt.ylabel('')
        plt.title(f'Matriz de confusión {classifier}')
    plt.show()


def compare_regression(y_predictions):
    """
    Recibe un diccionario con las etiquetas predecidas de cada algoritmo y crea una imagen para comparar 20 valores
    estimados aleatorios entre todos los algoritmos.

    Args:
        y_predictions (dict): Diccionario cuyas claves representan algoritmos y cuyos valores contienen etiquetas predecidas
    """
    # Lista de marcadores
    markers = ['bo', 'gd', 'y^', 'ms', 'cX', 'kp']

    # Crear nueva ventana
    plt.figure()
    for i, regressor in enumerate(y_predictions.keys()):
        if regressor == 'VottingRegressor':
            marker = 'r*'  # Si el regresor es VottingRegressor, asignar marcador de estrella roja
        else:
            marker = markers[i]  # En caso contrario, aplicar marcador de la lista

        # Dibujar 20 valores predecidos aleatorios de cada regresor
        plt.plot(y_predictions[regressor].sample(20, random_state=0), marker, label=regressor)

    # Aplicar parámetros de la imagen y mostrar
    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    plt.ylabel("Predicciones")
    plt.xlabel("Ejemplos aleatorios")
    plt.legend(loc="best")
    plt.title("Comparación de predicciones entre algoritmos de regresión")
    plt.show()


def plot_silhouette(X):
    """
    Encuentra el número óptimo de clústeres mediante el método del coeficiente de la silueta.

    Dado un conjunto de datos y con un rango preestablecido de k entre 2 y 10, calcula el coeficiente de la silueta
    con las etiquetas predecidas por el algoritmo KMeans. Dichas puntuaciones se almacenan para dibujar un gráfico en
    el que para cada valor de k en el eje X, se sitúa su correspondiente coeficiente de la silueta en el eje Y.
    Finalmente, el número óptimo de clústeres será aquel con mayor coeficiente de la silueta, por lo que se devuelven
    estos dos valores.

    Args:
        X (pd.DataFrame): DataFrame con el conjunto de datos sobre el que aplicar el método de optimización

    Returns:
        tuple(float, int): Valor máximo del coeficiente de la silueta y número óptimo de clústeres
    """
    silhouette_scores = []  # Puntuaciones de coeficiente de silueta
    k_values = range(2, 11)  # Rango de valores de k

    # Calcular coeficiente de la silueta para cada valor de k en KMeans
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))

    # Generar gráfico de coeficientes de silueta según cada valor de k
    plt.figure()
    plt.plot(k_values, silhouette_scores, 'bo-')
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Coeficiente de silueta')

    # Obtener máximo coeficiente de silueta y su número óptimo de clústeres k
    max_silhouette = max(silhouette_scores)
    max_index = silhouette_scores.index(max_silhouette) + 2

    return max_silhouette, max_index


def plot_dendrogram(model, **kwargs):
    """
    Encuentra el número óptimo de clústeres mediante un dendrograma con clustering jerárquico aglomerativo.

    Args:
        model (object): Model de clustering jerárquico aglomerativo de scikit-learn
        **kwargs: Argumentos adicionales para la función dendrogram de scipy.

    Returns:
        tuple(float, int): Diferencia máxima de distancias entre dos nodos y número óptimo de clústeres
    """
    # Crear matriz de vinculación y luego trazar el dendrograma
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # nodo hoja
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    Z = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Generar el correspondiente dendrograma
    plt.figure()
    dendrogram(Z, **kwargs)
    plt.xlabel('Número de nodos bajo la hoja o índice del nodo si no hay paréntesis')
    plt.xlabel('Distancia')

    # Calcular la mayor diferencia entre distancias consecutivas
    diffs = np.diff(np.sort(model.distances_))
    max_distance = np.max(diffs)

    # Dibujar una línea horizontal en el dendrograma en max_distance
    plt.axhline(y=max_distance + 0.05, color='k', linestyle='--', label=f'Distancia máxima: {max_distance:.2f}')
    plt.legend()

    # Calcular la distancia máxima
    num_clusters = len(np.unique(fcluster(Z, max_distance, criterion='distance')))

    return max_distance, num_clusters


def cluster_plots(X_pca, X_tsne, rows, cols, titles, labels_list):
    """
    Genera gráficos de dispersión utilizando técnicas de reducción de dimensionalidad PCA y t-SNE
    para visualizar clústeres según listas de etiquetas.

    Args:
        X_pca (ndarray): Matriz de datos transformados por PCA en 2D.
        X_tsne (ndarray): Matriz de datos transformados por t-SNE en 2D.
        rows (int): Número de filas acorde al número de algoritmos de clustering.
        cols (int): Número de columnas acorde al número de tipos de reducción de dimensionalidad.
        titles (list): Lista de títulos para cada gráfico.
        labels_list (list): Lista de etiquetas de clustering para cada gráfico.
    """
    # Crear una figura con gráficos según el número de filas y columnas
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

    # Crear una paleta de colores según clusters y asociar con colores
    unique_labels = set(label for labels in labels_list for label in np.unique(labels))
    num_labels = len(unique_labels)
    color_palette = px.colors.qualitative.Plotly[:num_labels]
    label_to_color = {label: color_palette[i] for i, label in enumerate(sorted(unique_labels))}

    # Generar las posiciones de los gráficos
    positions = [(i, j) for i in range(1, rows + 1) for j in range(1, cols + 1)]

    # Generar un rango para iterar sobre los gráficos
    count = range(rows * cols)

    # Iterar sobre las posiciones de los gráficos, el contador y la lista de etiquetas
    for (i, j), k, labels in zip(positions, count, labels_list):
        if k % 2 == 0:  # Si el índice es par, se usa la reducción PCA
            scatter = go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                                 mode='markers',
                                 marker=dict(color=[label_to_color[label] for label in labels],
                                             size=5))
            fig.add_trace(scatter, row=i, col=j)  # Agregar el gráfico a la figura
        else:  # Si el índice es impar, se usa la reducción t-SNE
            scatter = go.Scatter(x=X_tsne[:, 0], y=X_tsne[:, 1],
                                 mode='markers',
                                 marker=dict(color=[label_to_color[label] for label in labels],
                                             size=5))
            fig.add_trace(scatter, row=i, col=j)  # Agregar el gráfico a la figura

    # Mostrar el conjunto de gráficos completo
    fig.show()
