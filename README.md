# Análisis descriptivo y predictivo de características de asteroides y sus trayectorias

El presente repositorio contiene el código fuente en los lenguajes Python y R del TFG titulado "Análisis descriptivo y predictivo de características de asteroides y sus trayectorias", realizado por el alumno Samuel Sánchez Carrasco, estudiante en el Grado de Ingeniería informática en la Universidad de Jaén.

## Estructura

- **`Python/`**: Carpeta que contiene el código fuente de los algoritmos ML y los análisis de descripción y predicción.
    - **`datasets.zip`**: Archivo comprimido que almacena los conjuntos de datos preprocesados para cada análisis.
      -   **`asteroids_clas.csv`**: Dataset preprocesado para el análisis de predictivo de las trayectorias u órbitas (clasificación).
      -   **`asteroids_clus.csv`**: Dataset preprocesado para el análisis descriptivo de taxonomías espectrales (agrupamiento).
      -   **`asteroids_ra.csv`**: Dataset preprocesado para el análisis descriptivo de asteroides (reglas de asociación).
      -   **`asteroids_regr.csv`**: Dataset preprocesado para el análisis predictivo del tamaño (regresión).
    -   **`analisis.py`**: Archivo que contiene las diferentes funciones implementadas para cargar datos, calcular métricas o mostrar resultados.
    -   **`clasificacion.py`**: Archivo que contiene el análisis de clasificación.
    -   **`clustering.py`**: Archivo que contiene el análisis de agrupamiento.
    -   **`main.exe`**: Archivo ejecutable que ejecuta el código fuente por línea de comandos.
    -   **`main.py`**: Archivo principal de ejecución, que muestra interfaz por consola para interactuar con los métodos.
    -   **`reglas_asociacion.py`**: Archivo que contiene el análisis de reglas de asociación.
    -   **`regresion.py`**: Archivo que contiene el análisis de regresión.
    -   **`requirements.txt`**: Archivo de texto con las dependencias necesarias para ejecutar el código fuente.
-   **`R/`**: Carpeta que contiene el código fuente del análisis exploratorio y los preprocesamientos aplicados.
    -   **`asteroids_initial.zip`**: Archivo comprimido que almacena el conjunto de datos inicial seleccionado.
    -   **`analisisExploratorio.R`**: Archivo que contiene las funciones del análisis exploratorio.
    -   **`imports.R`**: Archivo que contiene los paquetes necesarios para aplicar el análisis exploratorio y preprocesamiento.
    -   **`preprocesamiento.R`**: Archivo que contiene las funciones del preprocesamiento genérico.
    -   **`preprocesamientoCLUS.R`**: Archivo que contiene las funciones del preprocesamiento para agrupamiento.
    -   **`preprocesamientoPRED.R`**: Archivo que contiene las funciones del preprocesamiento para regresión y clasificación.
    -   **`preprocesamientoRA.R`**: Archivo que contiene las funciones del preprocesamiento para reglas de asociación.
-   **`README.md`**: Documento que describe el repositorio y explica su modo de uso e instalación.

## Uso e instalación

-  Para aplicar el análisis exploratorio y preprocesamiento en el orden indicado para generar los conjuntos de datos preprocesados, primero descomprimir el archivo **`asteroids_initial.zip`** y asegurarse de tener instalado el lenguaje **`R v4.4.1`**. Se ejecutan secuencialmente cada una de las sentencias de **`imports.R`**, **`analisisExploratorio.R`**, **`preprocesamiento.R`**, **`preprocesamientoPRED.R`**, **`preprocesamientoRA.R`** y **`preprocesamientoCLUS.R`** en ese orden. Los archivos de R y el dataset deben compartir la misma carpeta de trabajo.

-  Para ejecutar **`main.exe`**, descomprimir el archivo **`datasets.zip`** en una carpeta del mismo nombre y situarla en el mismo directorio que el ejecutable.

-  En caso de querer probar el código fuente en un entorno Python, primero asegurarse de tener instalado **`Python v3.12`**. Instalar dependencias con **`pip install -r requirements.txt`**, descomprimir el archivo **`datasets.zip`** en una carpeta del mismo nombre y situarla en el mismo directorio que el código fuente, y ejecutar el comando **`python main.py`**.

## Contacto
Para dudas o sugerencias, contactar a [ssc00022@red.ujaen.es].