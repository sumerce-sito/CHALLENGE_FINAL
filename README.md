# **CHALLENGE FINAL**
Análisis y modelos de machine learning para predecir la cancelación de clientes en Telecom X, utilizando un conjunto de datos realista y técnicas de limpieza, EDA y clasificación.

Análisis y Predicción de Churn de Clientes
Este proyecto tiene como objetivo analizar los factores que influyen en la cancelación (churn) de clientes y desarrollar modelos predictivos para identificar a aquellos con mayor riesgo de irse.

Propósito del Análisis
El objetivo principal de este análisis es predecir el churn de clientes basándose en un conjunto de variables relevantes. La identificación temprana de clientes propensos a la cancelación permite implementar estrategias de retención proactivas.

**Estructura del Proyecto**
El proyecto está organizado de la siguiente manera:

nombre_del_cuaderno.ipynb: Cuaderno principal de Google Colab que contiene todo el código para el análisis, preprocesamiento, modelado y evaluación.
df_limpio.csv: Archivo CSV que contiene los datos tratados y listos para ser utilizados en el análisis.
visualizaciones/ (Opcional): Carpeta que podría contener visualizaciones generadas durante el Análisis Exploratorio de Datos (EDA) si se exportaron como archivos de imagen.

##**Preparación de los Datos**
El proceso de preparación de los datos incluyó las siguientes etapas:

  **Carga de Datos:** Se cargó el archivo CSV df_limpio.csv utilizando la librería Pandas.
  **Limpieza Inicial:** Se eliminaron columnas irrelevantes como customerID.
  **Manejo de Valores Especiales:** Se reemplazaron los valores 'No internet service' en las columnas relevantes por 'No'.
  **Manejo de Valores Nulos:** Se eliminaron las filas con valores nulos en las columnas Total.Day y account.Charges.Total.
  **Clasificación de Variables:** Las variables se clasificaron en categóricas y numéricas para aplicar el preprocesamiento adecuado.
  **Normalización:** Se aplicó StandardScaler a las columnas numéricas para escalarlas y prepararlas para modelos que requieren características en una escala similar.
  **Codificación:** Se aplicó One-Hot Encoding (pd.get_dummies()) a las variables categóricas para convertirlas en un formato numérico binario, adecuado para los modelos de machine learning. Se utilizó drop_first=True para evitar la multicolinealidad.
  **Balanceo de Clases:** Se identificó un desbalance en la proporción de clientes que cancelan (Churn). Para abordar esto, se aplicó la técnica SMOTE (Synthetic Minority Over-sampling Technique) al conjunto de entrenamiento para sobremuestrear la clase minoritaria (Churn = Yes).
  **División del Conjunto de Datos:** Los datos fueron divididos en conjuntos de entrenamiento y prueba utilizando train_test_split con una proporción de 70/30 y estratificación para mantener la proporción de clases en ambos conjuntos.

##**Modelado y Justificaciones**
Se entrenaron y evaluaron dos modelos predictivos: Regresión Logística y Random Forest.

Regresión Logística: Se eligió como un modelo lineal de referencia, fácil de interpretar y que proporciona información sobre la dirección e intensidad de la relación entre las características y la probabilidad de churn (a través de los coeficientes). Requiere características escaladas, lo cual se realizó en el paso de normalización. Se utilizó el solver='liblinear' por ser adecuado para conjuntos de datos pequeños y medianos.
Random Forest: Se eligió como un modelo de conjunto (ensemble) no lineal que generalmente tiene un buen rendimiento. Es menos sensible a la escala de las características (por lo que la normalización previa no es estrictamente necesaria para este modelo, pero el balanceo de clases sí es importante para manejar el desbalance).
Se realizó un análisis de multicolinealidad (VIF) para entender las relaciones entre las variables predictoras, aunque en este caso se decidió no eliminar variables basándose únicamente en VIF, ya que ambos modelos pueden manejar cierto grado de correlación (especialmente Random Forest) y el enfoque principal fue la predicción.

Se ajustaron los hiperparámetros del modelo Random Forest utilizando GridSearchCV para intentar reducir el overfitting y mejorar su rendimiento en datos no vistos, enfocándose en métricas como recall o f1 debido al desbalance de clases.

##**Análisis Exploratorio de Datos (EDA) e Insights**
Durante el EDA, se realizaron visualizaciones clave para entender la relación entre algunas variables y el churn:

Matriz de Correlación: Se visualizó una matriz de correlación para identificar las relaciones lineales entre todas las variables (incluyendo la variable objetivo 'Churn' convertida a numérica). Esto reveló qué variables tienen la correlación más fuerte (positiva o negativa) con la cancelación.
Boxplots: Se crearon boxplots para comparar la distribución de variables numéricas clave como customer.tenure y account.Charges.Total para los clientes que cancelaron y los que no. Estos gráficos ayudaron a visualizar si existen diferencias significativas en estas distribuciones entre las dos clases.
Los insights obtenidos de estos análisis, junto con la importancia de las variables de los modelos, informaron las conclusiones sobre qué factores influyen más en el churn y el perfil de cliente de alto riesgo.
