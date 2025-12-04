# -volatility-prediction-cnn
Proyecto de predicción binaria de volatilidad del S&amp;P 500 usando ventanas de 60 días de retornos logarítmicos y una CNN1D. Contiene limpieza, EDA, baseline y entrenamiento completo en PyTorch/fastai.

# Predicción de volatilidad del S&P 500 con CNN 1D

Este proyecto implementa un flujo completo para predecir periodos de **alta/baja volatilidad** en el índice S&P 500 usando **ventanas de 60 días de retornos logarítmicos** y una **red neuronal convolucional 1D (CNN1D)**.  

También se incluye un modelo **baseline** de regresión logística para comparar el desempeño.

---

## Estructura del repositorio

La estructura está organizada en dos carpetas principales:

```
/data
    S&P500_data.xlsx

/notebooks
    Limpieza_500.ipynb
    Analisis.ipynb
    baseline_stocks.ipynb
    Red_CNN1D_sotcks.ipynb
```

---


### Reporte del proyecto
El informe técnico final está disponible en la raíz del repositorio:

- **`Predicción_de_volatilidad_reporte.pdf`**

Este archivo contiene el desarrollo completo del proyecto, análisis, modelos y conclusiones.

## 📁 Carpeta `data/`

### `data/S&P500_data.xlsx`
Dataset **ya limpio**, generado desde el notebook de limpieza.  
Incluye:
- Precios ajustados del S&P 500  
- Retornos logarítmicos  
- Volatilidad futura (rolling std)  
- Etiqueta binaria (alta/baja volatilidad)  

Este archivo es utilizado por todos los notebooks posteriores.

---

## 📁 Carpeta `notebooks/`

Todos los notebooks se encuentran aquí.

### **1. `Limpieza_500.ipynb`**
Realiza el proceso de preparación del dataset:
- Descarga la serie histórica del S&P 500 desde Yahoo Finance  
- Calcula retornos logarítmicos y volatilidad futura  
- Genera la etiqueta binaria (alta/baja volatilidad)  
- Exporta el dataset final a `data/S&P500_data.xlsx`  

---

### **2. `Analisis.ipynb`**
Notebook de **análisis exploratorio (EDA)**:
- Visualización de retornos  
- Volatilidad rolling vs volatilidad futura  
- Boxplots e histogramas  
- Hallazgos clave sobre la dinámica de volatilidad

---

### **3. `baseline_stocks.ipynb`**
Implementación del **modelo baseline**:
- Usa `S&P500_data.xlsx`  
- Construye features simples (retornos rezagados y volatilidad rolling)  
- Entrena una regresión logística  
- Resultado:  
  - **Accuracy ≈ 64%**

Sirve como referencia para evaluar el modelo final.

---

### **4. `Red_CNN1D_sotcks.ipynb`**
Implementación del modelo final **CNN1D**:
- Genera secuencias de entrada de 60 días  
- Arquitectura:
```
- Conv1D → ReLU → MaxPool →
Conv1D → ReLU → MaxPool → Dropout →
AdaptiveAvgPool1D → Linear
```
- Entrenamiento con fastai:
- Adam (`lr = 1e-3`)
- Batch size 64  
- Dropout + weight decay  
- EarlyStopping  
- Resultados:
- **Accuracy ≈ 78–79%**

Incluye gráficas de training loss y validation loss.

### **5. `Flujo_completo.ipynb`**   
  Notebook que **integra todo el flujo del proyecto en un solo archivo**:
  - Carga del dataset limpio  
  - Análisis básico  
  - Preparación de secuencias  
  - Entrenamiento del baseline  
  - Entrenamiento de la CNN1D  
  - Curvas de loss y evaluación final  

  Este notebook permite **reproducir el proyecto completo desde cero de manera ordenada** y obtiene el mismo desempeño final (**≈79% accuracy**).

---

## Requisitos

Dependencias principales del proyecto:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `torch`
- `fastai`
- `yfinance`

Instalación:

```bash
pip install pandas numpy matplotlib scikit-learn torch fastai yfinance
