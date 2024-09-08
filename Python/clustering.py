"""
Nombre del proyecto: Análisis predictivo y descriptivo de características de asteroides y sus trayectorias
Archivo: clustering.py
Autor: Samuel Sánchez Carrasco
Fecha: 25/08/2024
Descripción: Análisis de clustering con búsqueda de número óptimo de clústeres con el método del coeficiente de la
             silueta y el dendrograma, predicción de clústeres con implementaciones de K-Means, Birch y DBSCAN
             y visualización en 2D de resultados mediante técnicas de reducción de dimensionalidad PCA y t-SNE.
"""

from analisis import load_dataset, plot_silhouette, plot_dendrogram, cluster_plots
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px


def clustering():
    """
    Del dataset 'asteroids_clus' se almacena externamente las etiquetas del atributo 'spec_B' para al final comparar
    con los resultados del clustering. Primero se ejecuta la búsqueda del número óptimo de clústeres mediante el
    método del coeficiente de la silueta, que devuelve tanto el valor máximo de la silueta como la gráfica en la que
    se visualizan los valores del coeficiente de silueta según cada k del algoritmo KMeans. De la misma forma,
    se hace lo propio con un dendrograma que utiliza un modelo AgglomerativeClustering para generarlo y que devuelva
    la mayor diferencia de distancias entre dos niveles. Una vez obtenidos los números de clústeres óptimos según
    cada método, los aplicamos en los algoritmos de clustering KMeans y Birch respectivamente. En el caso de DBSCAN
    no requiere un número de clústeres previo.

    Cada algoritmo devolverá las etiquetas predecidas de los clústeres. Dichas etiquetas servirán como entrada para
    los métodos de reducción de dimensionalidad PCA y t-SNE a 2D, de manera que se observarán a qué clúster pertenece
    cada muestra del conjunto de datos. Adicionalmente, observaremos las mismas gráficas con las etiquetas del atributo
    'spec_B' para comparar si existen ciertas similitudes con alguno de los algoritmos de clustering.
    """
    # Cargar DataFrame
    df = load_dataset('asteroids_clus')

    # Almacenar etiquetas de spec_B en estructura externa
    X = df.drop(columns=['spec_B'])
    spec_B = df['spec_B']

    # Buscar número óptimo de clústers mediante el método del coeficiente de silueta con KMeans
    silhouette, clusters_silhouette = plot_silhouette(X)
    print('\nMaximum silhouette with KMeans = ', silhouette)
    print('Number of clusters determined by silhouette = ', clusters_silhouette)

    # Buscar número óptimo de clústers mediante un dendrograma truncado a los tres primeros niveles
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X)
    distance, clusters_dendrogram = plot_dendrogram(model, truncate_mode="level", p=3)
    print('\nMaximum distance between nodes with Agglomerative Clustering = ', distance)
    print('Number of clusters determined by dendrogram = ', clusters_dendrogram, '\n')
    plt.show()

    # Predicción de clustering con KMeans
    labels_kmeans = KMeans(n_clusters=clusters_silhouette, random_state=0).fit_predict(X)

    # Predicción de clustering con Birch
    labels_birch = Birch(n_clusters=clusters_dendrogram).fit_predict(X)

    # Predicción de clustering con DBSCAN
    labels_dbscan = DBSCAN().fit_predict(X)

    # Almacenamos las etiquetas y los títulos de las gráficas
    labels_list = [labels_kmeans, labels_kmeans, labels_birch, labels_birch, labels_dbscan, labels_dbscan]
    titles = [f'Visualización PCA con K-means',
              f'Visualización t-SNE con K-means',
              f'Visualización PCA con Birch',
              f'Visualización t-SNE con Birch',
              f'Visualización PCA con DBSCAN',
              f'Visualización t-SNE con DBSCAN', ]

    # Reducción del espacio dimensional con PCA a 2D
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X)

    # Reducción del espacio dimensional con t-SNE a 2D
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)

    # Mostrar gráficas de clustering con etiquetas predecidas
    print('Printing PCA and t-SNE visualizations...')
    cluster_plots(X_pca, X_tsne, 3, 2, titles, labels_list)

    # Mostrar gráficas de clustering con etiquetas spec_B
    for data, method in zip([X_pca, X_tsne], ['PCA', 't-SNE']):
        fig = px.scatter(x=data[:, 0], y=data[:, 1], color=spec_B, title=f'Visualización {method} con etiquetas spec_B')
        fig.show()
