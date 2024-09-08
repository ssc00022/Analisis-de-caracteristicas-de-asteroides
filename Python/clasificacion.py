"""
Nombre del proyecto: Análisis predictivo y descriptivo de características de asteroides y sus trayectorias
Archivo: clasificacion.py
Autor: Samuel Sánchez Carrasco
Fecha: 25/08/2024
Descripción: Análisis de clasificación con implementaciones de KNN, NaiveBayes, SVM, DecisionTree y ANN a través de las
             métricas accuraccy, precision, recall, puntuación F1 y coeficiente de Kappa. Adicionalmente, se aporta
             matrices de confusión y reportes de clasificación para cada algoritmo.
"""

from analisis import load_dataset, prediction_metrics, show_tables, kappa_heatmap, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def classification():
    """
    Ejecuta la predicción del atributo 'class' del dataset 'asteroids_clas' con los algoritmos KNeighborsClassifier,
    GaussianNB, LinearSVC, DecisionTreeClassifier y MLPClassifier. La división del conjunto de entrenamiento y test
    se realiza mediante validación cruzada estratificada con 5 particiones y selección aleatoria.

    Las métricas medidas para este problema son Accuracy, Precision, Recall y F1-Score. Una vez acumulados los valores
    de las métricas de cada algoritmo en cada partición, se calcula el promedio de cada una de ellas y se muestran los
    resultados en la imagen de una tabla.

    Acumulando previamente las etiquetas predecidas en la validación cruzada y ordenándolas por orden de sus índices, se
    generan reportes de clasificación para cada algoritmo, un mapa de calor cuyos valores son coeficientes de Kappa para
    cada par de algoritmos y, finalmente, matrices de confusión.
    """
    # Cargar DataFrame
    df = load_dataset('asteroids_clas')

    # Separar atributos de la clase
    X = df.drop('class', axis=1)
    y = df['class']

    # Aplicar validación cruzada estratificada con 5 divisiones y selección aleatoria
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # Diccionario para almacenar las métricas para cada partición y clasificador
    metrics = {
        "KNeighborsClassifier": {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []},
        "GaussianNB": {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []},
        "LinearSVC": {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []},
        "DecisionTreeClassifier": {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []},
        "MLPClassifier": {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []}
    }

    # Diccionario para almacenar las etiquetas predecidas para cada partición y clasificador
    predictions = {
        "KNeighborsClassifier": [],
        "GaussianNB": [],
        "LinearSVC": [],
        "DecisionTreeClassifier": [],
        "MLPClassifier": []
    }
    y_index = []  # Orden de los índices según la validación cruzada

    # Iterar sobre cada partición
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        y_index.extend(test_index)
        print('Working on Fold ', i + 1, '...')

        # Clasificador KNN
        y_knn = KNeighborsClassifier().fit(X_train, y_train).predict(X_test)
        metrics = prediction_metrics(metrics, y_test, y_knn, True, 'KNeighborsClassifier')
        predictions['KNeighborsClassifier'].extend(y_knn)

        # Clasificador NaiveBayes
        y_gnb = GaussianNB().fit(X_train, y_train).predict(X_test)
        metrics = prediction_metrics(metrics, y_test, y_gnb, True, 'GaussianNB')
        predictions['GaussianNB'].extend(y_gnb)

        # Clasificador SVM
        y_svm = LinearSVC(random_state=0).fit(X_train, y_train).predict(X_test)
        metrics = prediction_metrics(metrics, y_test, y_svm, True, 'LinearSVC')
        predictions['LinearSVC'].extend(y_svm)

        # Clasificador Árbol de Decisión
        y_dtree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
        metrics = prediction_metrics(metrics, y_test, y_dtree, True, 'DecisionTreeClassifier')
        predictions['DecisionTreeClassifier'].extend(y_dtree)

        # Clasificador Red Neuronal
        y_mlp = MLPClassifier(random_state=0, max_iter=900).fit(X_train, y_train).predict(X_test)
        metrics = prediction_metrics(metrics, y_test, y_mlp, True, 'MLPClassifier')
        predictions['MLPClassifier'].extend(y_mlp)

        print(f"Fold {i + 1} complete")

    # Calcular la media de las métricas para cada clasificador
    average_metrics = {}
    for classifier in metrics.keys():
        average_metrics[classifier] = {
            "Accuracy": sum(metrics[classifier]["Accuracy"]) / len(metrics[classifier]["Accuracy"]),
            "Precision": sum(metrics[classifier]["Precision"]) / len(metrics[classifier]["Precision"]),
            "Recall": sum(metrics[classifier]["Recall"]) / len(metrics[classifier]["Recall"]),
            "F1-Score": sum(metrics[classifier]["F1-Score"]) / len(metrics[classifier]["F1-Score"]),
        }

    # Crear un DataFrame para mostrar los resultados promedio de las métricas
    print('Printing mean metrics for each algorithm...')
    mean_metrics = pd.DataFrame(average_metrics).transpose()
    show_tables([{'Análisis predictivo de las trayectorias u órbitas': mean_metrics}])

    y_pred_clas = {}  # Diccionario para almacenar las predicciones ordenadas por cada algoritmo
    reports = []  # Lista para almacenar reportes de clasificación de cada algoritmo
    conf_matrix = {}  # Diccionario para almacenar matrices de confusión de cada algoritmo
    class_labels = y.unique()  # Lista para almacenar clases únicas

    # Iterar sobre cada clasificador
    for classifier in predictions.keys():
        # Rellenar estructuras utilizando las predicciones y sus respectivos índices
        y_pred_clas[classifier] = pd.Series(data=predictions[classifier], index=y_index).sort_index()
        reports.append({f'Reporte de clasificación del algoritmo {classifier}':
                            pd.DataFrame(
                                classification_report(y, y_pred_clas[classifier], output_dict=True, zero_division=0))
                       .round(5).transpose()})
        conf_matrix[classifier] = confusion_matrix(y, y_pred_clas[classifier], labels=class_labels)

    # Generar tablas con reportes de clasificación de cada algoritmo
    print('Printing classification reports for each algorithm...')
    show_tables(reports)

    # Generar mapa de calor con coeficientes kappa de Cohen para cada par de algoritmos
    print('Generating kappa heatmap...')
    kappa_heatmap(y_pred_clas)

    # Generar matrices de confusión de cada algoritmo
    print('Generating confusion matrix for each algorithm...')
    plot_confusion_matrix(conf_matrix, class_labels)
