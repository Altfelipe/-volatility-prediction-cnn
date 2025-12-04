# -volatility-prediction-cnn
Proyecto de predicción binaria de volatilidad del S&amp;P 500 usando ventanas de 60 días de retornos logarítmicos y una CNN1D. Contiene limpieza, EDA, baseline y entrenamiento completo en PyTorch/fastai.

# Predicción de volatilidad del S&P 500 con CNN 1D

Este proyecto implementa un flujo completo para predecir periodos de **alta/baja volatilidad** en el índice S&P 500 usando **ventanas de 60 días de retornos logarítmicos** y una **red neuronal convolucional 1D (CNN1D)**.  

También se incluye un modelo **baseline** de regresión logística para comparar el desempeño.

---

## Estructura del repositorio

- `S&P500_data.xlsx`  
  Dataset **ya limpio** con precios diarios del S&P 500 y las columnas necesarias para los modelos.  
  Este archivo es el que usan todos los notebooks *después* de la etapa de limpieza.

- `Limpieza_500.ipynb`  
  Notebook donde:
  - Se descarga la serie histórica del S&P 500 desde **Yahoo Finance**.
  - Se calculan retornos logarítmicos, volatilidad futura (rolling std) y la variable objetivo (label alta/baja volatilidad).
  - Se guarda el resultado final en `S&P500_data.xlsx`.

- `Analisis (1).ipynb`  
  Notebook de **análisis exploratorio**, donde se trabaja con `S&P500_data.xlsx`:
  - Visualización de retornos logarítmicos.
  - Volatilidad rolling y volatilidad futura.
  - Boxplots, clustering de volatilidad y hallazgos descriptivos.

- `baseline_stocks.ipynb`  
  Notebook del **modelo baseline**:
  - Carga `S&P500_data.xlsx`.
  - Construye features simples (retornos rezagados, volatilidad rolling).
  - Entrena una **regresión logística** para clasificar volatilidad alta/baja.
  - Reporta métricas (accuracy ≈ 64%) y sirve como punto de referencia.

- `Red_CNN1D_sotcks.ipynb`  
  Notebook del **modelo final CNN1D**:
  - Carga `S&P500_data.xlsx`.
  - Genera las secuencias de entrada de 60 días para la red (shape `(n_samples, 1, 60)`).
  - Define y entrena la red CNN1D en PyTorch/fastai con:
    - Conv1D → ReLU → MaxPool → Conv1D → ReLU → MaxPool → Dropout → AdaptiveAvgPool1D → Linear
  - Usa Adam, `lr = 1e-3`, batch size 64, Dropout + weight decay y EarlyStopping.
  - Reporta las curvas de loss (train/valid) y el desempeño final (accuracy ≈ 78–79%).

---

## Requisitos

- Python 3.x  
- Librerías principales:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `torch`
  - `fastai`
  - `yfinance` (para descargar datos en el notebook de limpieza)

Puedes instalar (de ejemplo):

```bash
pip install pandas numpy matplotlib scikit-learn torch fastai yfinance
