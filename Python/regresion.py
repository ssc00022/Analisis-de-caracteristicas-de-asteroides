"""
Nombre del proyecto: Análisis predictivo y descriptivo de características de asteroides y sus trayectorias
Archivo: regresion.py
Autor: Samuel Sánchez Carrasco
Fecha: 25/08/2024
Descripción: Análisis de regresión con implementaciones de regresión lineal, árbol de decisión y ensemble a
             través de las métricas de error cuadrático medio, raíz de error cuadrático medio, error absoluto medio, y
             puntuación R². También se incorporaan diagramas de dispersión entre valores reales y predecidos, además
             de una visualización comparativa entre predicciones aleatorias de los algoritmos.
"""
from analisis import load_dataset, prediction_metrics, show_tables, compare_regression
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, \
    VotingRegressor
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


def regression():
    """
    Ejecuta la predicción del atributo 'diameter' del dataset 'asteroids_regr' con los algoritmos LinearRegression,
    DecisionTreeRegressor, AdaboostRegressor, HistGradientBoostingRegressor, BaggingRegressor, RandomForestRegressor
    y VottingRegressor. La división del conjunto de entrenamiento y test se realiza mediante validación cruzada con 5
    particiones y selección aleatoria.

    Las métricas medidas para este problema son MSE, RMSE, MAE y R² Score. Una vez acumulados los valores de las métricas
    de cada algoritmo en cada partición, se calcula el promedio de cada una de ellas y se muestran los resultados en
    la imagen de una tabla.

    Acumulando previamente las etiquetas predecidas en la validación cruzada y ordenándolas por orden de sus índices, se
    generan diagramas de dispersión entre los valores reales de las etiquetas y los valores predecidos por cada algoritmo.
    Además, se generará una imagen comparativa que muestra para un númmero de ejemplos aleatorio los resultados predecidos
    y comparándose entre los algoritmos con un estilo muy visual y llamativo.
    """
    # Cargar DataFrame
    df = load_dataset('asteroids_regr')

    # Separar atributos de la clase
    X = df.drop('diameter', axis=1)
    y = df['diameter']

    # Aplicar validación cruzada con 5 divisiones y selección aleatoria
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # Diccionario para almacenar las métricas para cada partición y modelo de regresión
    metrics = {
        "LinearRegression": {"MSE": [], "RMSE": [], "MAE": [], "R² Score": []},
        "DecisionTree": {"MSE": [], "RMSE": [], "MAE": [], "R² Score": []},
        "Adaboost": {"MSE": [], "RMSE": [], "MAE": [], "R² Score": []},
        "HistGradientBoost": {"MSE": [], "RMSE": [], "MAE": [], "R² Score": []},
        "Bagging": {"MSE": [], "RMSE": [], "MAE": [], "R² Score": []},
        "RandomForest": {"MSE": [], "RMSE": [], "MAE": [], "R² Score": []},
        "VottingRegressor": {"MSE": [], "RMSE": [], "MAE": [], "R² Score": []},
    }

    # Diccionario para almacenar los valores predecidos para cada partición y modelo de regresión
    predictions = {
        "LinearRegression": [],
        "DecisionTree": [],
        "Adaboost": [],
        "HistGradientBoost": [],
        "Bagging": [],
        "RandomForest": [],
        "VottingRegressor": []
    }
    y_index = []  # Orden de los índices según la validación cruzada

    # Iterar sobre cada partición
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        y_index.extend(test_index)
        print('Working on Fold ', i + 1, '...')

        # Métodos de regresión
        lr = LinearRegression()
        dtree = DecisionTreeRegressor(random_state=0)
        ada = AdaBoostRegressor(random_state=0)
        hgb = HistGradientBoostingRegressor(random_state=0)
        bagg = BaggingRegressor(random_state=0)
        rf = RandomForestRegressor(random_state=0)

        # VottingRegressor
        vr = (VotingRegressor([("lr", lr), ("dtree", dtree), ("ada", ada), ("hgb", hgb), ("bagg", bagg), ("rf", rf)])
              .fit(X_train, y_train))
        estimators = vr.estimators_
        y_vr = vr.predict(X_test)
        metrics = prediction_metrics(metrics, y_test, y_vr, False, 'VottingRegressor')
        predictions['VottingRegressor'].extend(y_vr)

        # Regresión lineal
        y_lr = estimators[0].predict(X_test)
        metrics = prediction_metrics(metrics, y_test, y_lr, False, 'LinearRegression')
        predictions['LinearRegression'].extend(y_lr)

        # Árbol de decisión
        y_dtree = estimators[1].predict(X_test)
        metrics = prediction_metrics(metrics, y_test, y_dtree, False, 'DecisionTree')
        predictions['DecisionTree'].extend(y_dtree)

        # Adaboost
        y_ada = estimators[2].predict(X_test)
        metrics = prediction_metrics(metrics, y_test, y_ada, False, 'Adaboost')
        predictions['Adaboost'].extend(y_ada)

        # HistGradientBoosting
        y_hgb = estimators[3].predict(X_test)
        metrics = prediction_metrics(metrics, y_test, y_hgb, False, 'HistGradientBoost')
        predictions['HistGradientBoost'].extend(y_hgb)

        # Bagging
        y_bagg = estimators[4].predict(X_test)
        metrics = prediction_metrics(metrics, y_test, y_bagg, False, 'Bagging')
        predictions['Bagging'].extend(y_bagg)

        # RandomForest
        y_rf = estimators[5].predict(X_test)
        metrics = prediction_metrics(metrics, y_test, y_rf, False, 'RandomForest')
        predictions['RandomForest'].extend(y_rf)

        print(f"Fold {i + 1} complete")

    # Calcular la media de las métricas para cada modelo de regresión
    average_metrics = {}
    for regressor in metrics.keys():
        average_metrics[regressor] = {
            "MSE": sum(metrics[regressor]["MSE"]) / len(metrics[regressor]["MSE"]),
            "RMSE": sum(metrics[regressor]["RMSE"]) / len(metrics[regressor]["RMSE"]),
            "MAE": sum(metrics[regressor]["MAE"]) / len(metrics[regressor]["MAE"]),
            "R² Score": sum(metrics[regressor]["R² Score"]) / len(metrics[regressor]["R² Score"])
        }

    # Crear un DataFrame para mostrar los resultados
    print('Printing mean metrics for each algorithm...')
    results = pd.DataFrame(average_metrics).transpose()
    show_tables([{'Análisis predictivo del tamaño': results}])

    y_pred_regr = {}  # Diccionario para almacenar las predicciones ordenadas por cada algoritmo

    # Iterar sobre cada modelo de regresión
    for regressor in predictions.keys():
        # Almacenar y ordenar predicciones según sus respectivos índices
        y_pred_regr[regressor] = pd.Series(data=predictions[regressor], index=y_index).sort_index()
        # Crear diagramas de dispersión entre etiquetas reales y etiquetas predecidas
        PredictionErrorDisplay.from_predictions(y_true=y, y_pred=y_pred_regr[regressor], kind='actual_vs_predicted')
        plt.title(regressor)
    print('Printing prediction error displays for each algorithm...')
    plt.show()

    # Comparar 20 predicciones aleatorias entre todos los algoritmos
    print('Printing classification reports for each algorithm...')
    compare_regression(y_pred_regr)
