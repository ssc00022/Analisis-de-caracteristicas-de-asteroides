"""
Nombre del proyecto: Análisis predictivo y descriptivo de características de asteroides y sus trayectorias
Archivo: main.py
Autor: Samuel Sánchez Carrasco
Fecha: 25/08/2024
Descripción: Menú principal por consola para ejecutar el análisis deseado
"""

from clasificacion import classification
from regresion import regression
from reglas_asociacion import reglas_asociacion
from clustering import clustering

repeat = "y"

while repeat == "y":
    # Elegir un análisis para visualizar resultados
    print("\nElige un análisis de los siguientes (escribe el número):\n"
          "\t1. Análisis predictivo de las trayectorias u órbitas\n"
          "\t2. Análisis predictivo del tamaño\n"
          "\t3. Análisis descriptivo de asteroides\n"
          "\t4. Análisis descriptivo de taxonomías espectrales\n")
    analysis = input('>> ')

    if analysis == '1':
        classification()
    if analysis == '2':
        regression()
    if analysis == '3':
        reglas_asociacion()
    if analysis == '4':
        clustering()

    # Repetir proceso
    print("\n¿Desea ejecutar otro análisis? (y / n)")
    repeat = input('>> ')
