# Proyecto Final: Hiperheurística de Selección para el Problema del Viajante (TSP)

## 1. Descripción del proyecto

Este proyecto implementa una hiperheurística de selección basada en aprendizaje automático para el **Problema del Viajante (Traveling Salesman Problem, TSP)**.

El objetivo es desarrollar un modelo capaz de seleccionar, a partir de las características de una instancia, cuál de tres algoritmos heurísticos es el más conveniente para obtener una ruta de menor costo.

Se comparan los siguientes algoritmos:

- **2-opt Hill Climbing:** utiliza búsqueda local para mejorar una ruta mediante intercambios de aristas.
- **SA + 2-opt:** utiliza Simulated Annealing combinado con movimientos 2-opt para explorar soluciones y evitar quedar atrapado en óptimos locales.
- **MST + 2-opt:** construye una solución inicial basada en un árbol de expansión mínima y posteriormente la mejora mediante 2-opt.

Los resultados de las ejecuciones se utilizan para generar etiquetas que permiten entrenar un clasificador Random Forest.

## 2. Estructura del proyecto

La estructura esperada del repositorio es la siguiente:

```text
tsp_dataset/
│
├── data/
│   ├── instances/
│   │   ├── inst_0001.csv
│   │   ├── inst_0002.csv
│   │   └── ...
│   │
│   ├── instructor_reference/
│   │
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── main.py
│   ├── generate_labels.py
│   └── compare_models.py
│
├── results/
│   └── data/
│   │   ├── generate_dataset.py
│   │   ├── make_student_csvs.py
│   │   ├── requirements.txt
│   │   └── README.md
```

**Descripción de los archivos principales:**

| Archivo | Función |
|---|---|
| `generate_dataset.py` | Genera las instancias del problema TSP. |
| `make_student_csvs.py` | Prepara los conjuntos de entrenamiento y prueba. |
| `src/generate_labels.py` | Ejecuta los algoritmos y genera los costos y etiquetas. |
| `src/main.py` | Entrena y evalúa el modelo Random Forest final. |
| `src/compare_models.py` | Compara distintos modelos de clasificación mediante validación cruzada. |

## 3. Requisitos

- Python 3.10 o superior.
- pip.
- NumPy.
- Pandas.
- Scikit-learn.
- Matplotlib.

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Si no se cuenta con un archivo `requirements.txt`, se pueden instalar directamente:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

En Windows también se puede utilizar:

```powershell
py -m pip install numpy pandas scikit-learn matplotlib
```

## 4. Reproducción del proyecto

Todos los comandos deben ejecutarse desde la carpeta raíz del repositorio.

### Paso 1. Generar las instancias TSP

Ejecutar:

```bash
python generate_dataset.py
```

Este script genera las instancias utilizando una semilla fija (`SEED = 42`) para permitir la reproducibilidad de los datos.

Las instancias se guardan en:

```text
data/instances/
```

El conjunto contempla 500 instancias con diferentes características y distribuciones espaciales.

### Paso 2. Preparar los conjuntos de entrenamiento y prueba

Ejecutar:

```bash
python make_student_csvs.py
```

Este script prepara los archivos que se utilizarán durante la experimentación.

Los archivos resultantes son:

```text
data/train.csv
data/test.csv
```

La distribución utilizada es:

- Entrenamiento: 351 instancias (70.2%).
- Prueba: 149 instancias (29.8%).

Estos archivos contienen los identificadores de las instancias y sus características, como el número de ciudades, la distribución espacial y las medidas de distancia.

### Paso 3. Ejecutar los algoritmos y generar etiquetas

Ejecutar:

```bash
python src/generate_labels.py
```

El programa procesa las instancias de entrenamiento y prueba, ejecuta los tres algoritmos y registra el costo obtenido por cada uno.

Para cada instancia se generan las siguientes columnas:

- `cost_2opt_hill_climbing`
- `cost_sa_2opt`
- `cost_mst_2opt`
- `best_algorithm`

La columna `best_algorithm` identifica el algoritmo que obtuvo el menor costo en las ejecuciones realizadas.

Los resultados se guardan en:

```text
results/data/train_labeled.csv
results/data/test_labeled.csv
```

**Importante:** las etiquetas se generan a partir de los costos experimentales de los tres algoritmos. Por lo tanto, este paso debe ejecutarse antes de entrenar el modelo.

### Paso 4. Entrenar y evaluar Random Forest

Ejecutar:

```bash
python src/main.py
```

Este script carga los conjuntos etiquetados, entrena el clasificador Random Forest y realiza las predicciones sobre el conjunto de prueba.

El modelo final utiliza los siguientes parámetros:

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_features="sqrt"
)
```

Las características utilizadas para entrenar el modelo son:

```python
features = [
    "mean_pairwise_dist",
    "std_coords",
    "mean_centroid_dist",
    "mean_nn_dist"
]
```

La variable objetivo es:

```python
target = "best_algorithm"
```

El programa permite evaluar el desempeño del clasificador mediante métricas como:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Matriz de confusión.

También permite comparar el resultado del clasificador con el baseline, que consiste en predecir siempre el algoritmo más frecuente en los datos de entrenamiento.

### Paso 5. Comparar los modelos de clasificación

Para reproducir la comparación de los modelos utilizados durante la experimentación, ejecutar:

```bash
python src/compare_models.py
```

Se evaluaron diferentes clasificadores:

- Random Forest.
- Random Forest con balanceo de clases.
- Extra Trees.
- Decision Tree.
- Support Vector Machine (SVM).

La comparación se realizó mediante validación cruzada estratificada de cinco particiones (`StratifiedKFold`), utilizando Accuracy y F1 macro.

El modelo final seleccionado fue Random Forest, con 300 árboles y `max_features="sqrt"`.

## 5. Resultados experimentales

El conjunto completo contiene 500 instancias. La distribución de las etiquetas obtenidas fue:

| Algoritmo ganador | Instancias | Porcentaje |
|---|---:|---:|
| 2-opt Hill Climbing | 323 | 64.60% |
| SA + 2-opt | 92 | 18.40% |
| MST + 2-opt | 85 | 17.00% |
| **Total** | **500** | **100%** |

### Desempeño del modelo final

En el conjunto de prueba, el Random Forest original obtuvo:

| Métrica | Resultado |
|---|---:|
| Accuracy | 61.74% |
| F1 macro | 0.42 |

El baseline obtuvo un Accuracy de 69.13% y un F1 macro de aproximadamente 0.2725.

Aunque el Random Forest no superó al baseline en Accuracy, consiguió un F1 macro superior, lo que indica que identifica algunas instancias en las que los algoritmos minoritarios son mejores.

## 6. Consideraciones sobre la reproducibilidad

Para obtener resultados comparables es importante:

1. Utilizar la misma semilla de generación de instancias.
2. Mantener los archivos de configuración de los algoritmos.
3. No modificar los parámetros de las heurísticas entre ejecuciones.
4. Utilizar los mismos conjuntos de entrenamiento y prueba.
5. Mantener los parámetros del Random Forest final.
6. Ejecutar los scripts en el orden indicado.

Los resultados pueden variar si se modifican los parámetros de los algoritmos, las semillas aleatorias, las características de entrada o las versiones de las bibliotecas.

## 7. Tecnologías utilizadas

- **Python:** lenguaje de programación principal.
- **NumPy:** operaciones numéricas.
- **Pandas:** manipulación de datos y archivos CSV.
- **Scikit-learn:** entrenamiento y evaluación de modelos de aprendizaje automático.
- **Matplotlib:** visualización de resultados.

## 8. Autores

Proyecto académico de implementación de una hiperheurística de selección para el Problema del Viajante (TSP).
