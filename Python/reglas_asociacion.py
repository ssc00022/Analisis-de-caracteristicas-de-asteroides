"""
Nombre del proyecto: Análisis predictivo y descriptivo de características de asteroides y sus trayectorias
Archivo: reglas_asociacion.py
Autor: Samuel Sánchez Carrasco
Fecha: 25/08/2024
Descripción: Análisis de reglas de asociación con implementaciones de Apriori y FP-Growth evaluadas a través de tablas
             con conjuntos de datos más frecuentes y reglas, empleando métricas de soporte y confianza.
"""

from analisis import load_dataset, show_tables
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import pandas as pd


def reglas_asociacion():
    """
    Convierte el dataset 'asteroids_ra' primero en una lista de transacciones y después en una matriz de
    transacciones binaria sobre la que se aplican los algoritmos Apriori y FP-Growth para obtener los conjuntos de
    items más frecuentes. Dichos conjuntos serán la entrada para generar reglas de asociación que aportarán los
    antecedentes y consecuentes evaluados con las métricas de soporte y confianza. Finalmente, todas las tablas se
    mostrarán a través de imágenes.
    """
    # Cargar DataFrame
    df = load_dataset('asteroids_ra')

    # Transformar dataset en una lista de transacciones convirtiendo cada fila en una transacción
    transactions = []
    for index, row in df.iterrows():
        transaction = []
        for col in df.columns:
            transaction.append(f"{col}={row[col]}")
        transactions.append(transaction)

    # Convertir datos a una matriz de transacciones binarias
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_binario = pd.DataFrame(te_ary, columns=te.columns_)

    # Apriori
    frequent_itemsets_apriori = apriori(df_binario, min_support=0.1, use_colnames=True)  # Encontrar itemsets frecuentes
    frequent_itemsets_apriori['itemsets'] = (frequent_itemsets_apriori['itemsets'].apply(lambda x: list(x)))
    rules_apriori = association_rules(frequent_itemsets_apriori)  # Genera reglas de asociación
    rules_apriori = rules_apriori[['antecedents', 'consequents', 'support', 'confidence']]
    rules_apriori['antecedents'] = (rules_apriori['antecedents'].apply(lambda x: list(x)))
    rules_apriori['consequents'] = (rules_apriori['consequents'].apply(lambda x: list(x)))
    print('Printing association rules for Apriori...')

    # FP-Growth
    frequent_itemsets_fpgrowth = fpgrowth(df_binario, min_support=0.1,
                                          use_colnames=True)  # Encontrar itemsets frecuentes
    frequent_itemsets_fpgrowth['itemsets'] = (frequent_itemsets_fpgrowth['itemsets'].apply(lambda x: list(x)))
    rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth)  # Genera reglas de asociación
    rules_fpgrowth = rules_fpgrowth[['antecedents', 'consequents', 'support', 'confidence']]
    rules_fpgrowth['antecedents'] = (rules_fpgrowth['antecedents'].apply(lambda x: list(x)))
    rules_fpgrowth['consequents'] = (rules_fpgrowth['consequents'].apply(lambda x: list(x)))
    print('Printing association rules for FP-Growth...')

    # Mostrar tablas como imágenes
    results = [
        {'Conjuntos de items frecuentes (Apriori)': frequent_itemsets_apriori},
        {'Reglas de asociación (Apriori)': rules_apriori},
        {'Conjuntos de items frecuentes (FP-Growth)': frequent_itemsets_fpgrowth},
        {'Reglas de asociación (FP-Growth)': rules_fpgrowth},
    ]
    show_tables(results)
