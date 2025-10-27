# Predicción de situaciones de trata de personas (2020–2023) — README

> **Proyecto de Aprendizaje Automático** · Tecnicatura en Ciencia de Datos e IA  
> **Autora:** Ana María Fernández · **Ámbito:** Oficina de Rescate y Acompañamiento (AR)  
> **Enfoque territorial:** Tierra del Fuego (transferencia del modelo)

## 1) Objetivo
**Clasificar** si una intervención de la Oficina de Rescate (ene-2020 a ago-2023) corresponde a **trata (1)** o **no trata (0)**.

- **Objetivo general:** construir un clasificador binario robusto y transferible a contextos de baja frecuencia (Tierra del Fuego).
- **Objetivos específicos:**
  1. EDA y preparación (valores faltantes, desbalance, patrones regionales).
  2. Entrenar y comparar modelos supervisados priorizando **recall**.
  3. **Transfer learning local:** evaluar el mejor modelo nacional sobre los **27** casos de TDF.

## 2) Datos
- **Fuente:** `oficina-rescate-orientaciones-202001-202308.csv`
- **Registros:** **7.853** (Argentina).  
  Confirmadas como trata: **4.241** (~54%).  
  **Tierra del Fuego:** 27 intervenciones, 20 casos de trata.
- **Target:** `es_trata` (1/0)
- **Features relevantes (ejemplos):** `es_anonima`, `origen`, `subtema`, `provincia`, `via_ingreso`, `consultante_genero`, `consultante_edad_aparente`.

## 3) Metodología
- **Preprocesamiento:** imputación (KNNImputer), one-hot para categóricas, standard scaling para numéricas, split 80/20 estratificado.
- **Modelos evaluados:** Regresión Logística (baseline interpretable), Árbol de Decisión, Random Forest y **HistGradientBoostingClassifier** (como baseline alternativo).
- **Validación:** CV estratificada (k=5). **Optimización** por Grid/Random Search enfocada en **recall**.
- **Selección y umbral:** curva PR y ajuste de umbral para maximizar recall con costo controlado de FP.

## 4) Resultados (resumen)
- **Mejor pipeline:** `Tuned-LogisticRegression` con **umbral = 0.328** (optimizado por PR para recall temprano).
- **Baseline alternativo:** `HistGradientBoostingClassifier` con **F1 = 0.304** (referencia).
- **Archivos de respaldo de nulos:**  
  - `results/nulos_antes.csv`  
  - `results/nulos_despues.csv`
- **Métricas detalladas:** ver `results/metrics_*.csv` y gráficos en `figs/` (PR, ROC, matrices 0.5 vs óptimo, calibration).

> ⚠️ *Nota*: Las métricas finales exactas (precision, recall, F1, ROC-AUC) y las tablas por umbral deben tomarse **tal cual** del notebook del proyecto y ya están exportadas en `results/`. Este README **respeta** nombres, umbrales y valores clave indicados por la autora.


## 5) Transferencia a Tierra del Fuego
- Se aplica el mejor clasificador nacional (`Tuned-LogisticRegression @ thr=0.328`) al subconjunto TDF (27 casos).
- Se reporta desempeño local (confusión, recall, PPV) y se revisan errores FN/FP para **ajustes de umbral** y **criterios operativos**.

## 6) Estructura del repositorio (Cookiecutter Data Science)
```
├─ data/
│  ├─ raw/     # CSV original
│  ├─ interim/ # limpiezas parciales
│  └─ processed/ # dataset canónico
├─ notebooks/
│  ├─ 01_eda_preprocesamiento.ipynb
│  ├─ 02_modelado_cv_tuning.ipynb
│  ├─ 03_umbral_pr_y_diagnosticos.ipynb
│  └─ 04_transfer_tdf.ipynb
├─ results/
│  ├─ nulos_antes.csv
│  ├─ nulos_despues.csv
│  ├─ metrics_cv.csv
│  └─ metrics_test_thr0328.csv
├─ figs/
│  ├─ pr_curve_thr0328.png
│  ├─ roc_curve.png
│  ├─ confusion_050.png
│  └─ confusion_thr0328.png
└─ README.md
```

## 7) Consideraciones éticas y privacidad
- Anonimización estricta, no publicar PII. Uso educativo con fines de mejora operativa.

## 8) Entorno
- Python 3.10 · pandas 1.5 · numpy 1.23 · scikit-learn 1.2 · imbalanced-learn · shap · seaborn

## 9) Citas y marco de clase
- Clase 4: Regresión lineal y logística; Clase 5: KNN y Árboles; Clase 6: SVM/SGD; Clase 8: Clustering (material teórico y prácticas).

---

© 2025 Ana María Fernández — Tecnicatura en Ciencia de Datos e IA
