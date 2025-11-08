# Predicción de situaciones de trata de personas (2020–2024) — README

> **Proyecto de Aprendizaje Automático** · Tecnicatura en Ciencia de Datos e IA  
> **Autora:** Ana María Fernández · **Ámbito:** Oficina de Rescate y Acompañamiento (AR)  
> **Enfoque territorial:** Tierra del Fuego (transferencia del modelo)

## 1) Objetivo
**Clasificar** si una intervención de la Oficina de Rescate (ene-2020 a oct-2024) corresponde a **trata (1)** o **no trata (0)**.

- **Objetivo general:** construir un clasificador binario robusto y transferible a contextos de baja frecuencia (Tierra del Fuego).
- **Objetivos específicos:**
  1. EDA y preparación (valores faltantes, balance, patrones regionales).
  2. Entrenar y comparar modelos supervisados priorizando **recall**.
  3. **Transferencia local (TDF):** evaluar el mejor modelo nacional sobre el subconjunto local (n pequeño) y ajustar umbral si es necesario.

📄 **Informe completo (PDF):** [Ver](https://drive.google.com/file/d/1AvKjNq2TPsjG6Hjy8Ap9MwXs9K5kZrEF/view?usp=sharing) · [Descargar](https://drive.google.com/uc?export=download&id=1AvKjNq2TPsjG6Hjy8Ap9MwXs9K5kZrEF)  
📁 **Carpeta del proyecto (Drive):** [Abrir](https://drive.google.com/drive/folders/1Pi_5rFwRCzmmJpSQl1gV6k_Ke6B7OvzF?usp=drive_link)

---

## 2) Datos
- **Fuente:** `oficina-rescate-orientaciones-202001-202308.csv` (2020–2024).
- **Registros (forma final):** **7.848** filas · **26** variables.  
- **Balance:** `es_trata=1` **54%** (4.241) / `0` **46%** (3.607).  
- **Target:** `es_trata` (1/0).
- **Principales transformaciones:** normalización de strings (lowercase/sin tildes), estandarización de provincia/localidad/nacionalidad, derivación temporal (año/mes/trimestre + sin/cos), banderas (`es_fin_semana`, `es_anonima`), uso de IDs geográficos cuando están disponibles.
- **Calidad de datos:** tablas de nulos antes/después en `results/nulos_antes.csv` y `results/nulos_despues.csv`.

---

## 3) Metodología
- **Validación:** split **temporal** (train/valid/test por fechas) sin fuga; **backtesting rolling-origin** mensual (2020-07→2024-12).
- **Optimización de umbral:** por **curva Precision–Recall** con restricción **recall ≥ 0.80**.
- **Calibración de probabilidades:** Isotónica/Platt; evaluación por **Brier** y curva de calibración.
- **Modelos evaluados:** Logistic Regression (base), **Logistic Regression + interacciones** (temporada×anonimato, provincia×anonimato, nacionalidad×temporada) y **HistGradientBoosting** (con/sin calibración).
- **Reproducibilidad:** pipelines y umbrales persistidos; semillas fijas.

---

## 4) Resultados (resumen)
- **Modelo seleccionado (operativo):** **Logistic Regression + interacciones** con **umbral = 0.345** (PR con recall ≥ 0.80).  
  **Test:** **Precision 0.563 · Recall 0.972 · F1 0.713 · ROC-AUC 0.623**.
- **Alternativa si se prioriza F1/ROC:** **HistGradientBoosting calibrado** con **umbral = 0.396**.  
  **Test:** **Precision 0.562 · Recall 0.958 · F1 0.708 · ROC-AUC 0.659 · AP 0.685**  
  (Brier **0.243 → 0.234** tras calibración).
- **Modelo base (referencia):** **Tuned-LogisticRegression @ thr = 0.328**.  
  **Test:** **Precision 0.559 · Recall 0.951 · F1 0.704 · AP 0.667 · ROC-AUC 0.628**.
- **Backtesting temporal (promedios):** **Precision 0.647 · Recall 0.686 · F1 0.651**.

**Archivos clave exportados**
- Métricas/tablas:  
  `results/modelos_metricas.csv`, `results/hp_search_resumen.csv`, `results/hp_best_holdout_metrics.csv`,  
  `results/best_metrics_Tuned-LogisticRegression_c16.csv`,  
  `results/best_threshold_Tuned-LogisticRegression_c16.json`,  
  `results/classification_report_Tuned-LogisticRegression_opt_c16.txt`
- Figuras (ejemplos):  
  `figs/pr_Tuned-LogisticRegression_c16.png`, `figs/pr_HGB.png`, `figs/roc_HGB.png`,  
  `figs/cm_Tuned-LogisticRegression_050_c16.png`, `figs/cm_Tuned-LogisticRegression_opt_c16.png`

---

## 5) Transferencia a Tierra del Fuego
- Evaluación del mejor clasificador nacional sobre **TDF** (muestra chica).  
- Con **LogReg + interacciones** y **mismo umbral (0.345)** en corrida específica (**n=30**, **positivos=22**):  
  **Precision 0.733 · Recall 1.00 · F1 0.846**.  
  *Cautela por bajo N; monitoreo mensual y recalibración si cambia la casuística.*

---

## 6) Estructura del repositorio
```
├─ data/
│  ├─ raw/         # CSV original
│  ├─ interim/     # limpiezas parciales
│  └─ processed/   # dataset canónico
├─ notebooks/
│  ├─ 01_eda_preprocesamiento.ipynb
│  ├─ 02_modelado_cv_tuning.ipynb
│  ├─ 03_umbral_pr_y_diagnosticos.ipynb
│  └─ 04_transfer_tdf.ipynb
├─ results/
│  ├─ nulos_antes.csv
│  ├─ nulos_despues.csv
│  ├─ modelos_metricas.csv
│  ├─ hp_search_resumen.csv
│  ├─ hp_best_holdout_metrics.csv
│  ├─ best_metrics_Tuned-LogisticRegression_c16.csv
│  ├─ best_threshold_Tuned-LogisticRegression_c16.json
│  └─ classification_report_Tuned-LogisticRegression_opt_c16.txt
├─ figs/
│  ├─ pr_Tuned-LogisticRegression_c16.png
│  ├─ pr_HGB.png
│  ├─ roc_HGB.png
│  ├─ cm_Tuned-LogisticRegression_050_c16.png
│  └─ cm_Tuned-LogisticRegression_opt_c16.png
└─ README.md
```

---

## 7) Consideraciones éticas y privacidad
- Anonimización estricta; no publicar PII. Uso educativo con orientación a mejora operativa.

## 8) Entorno
- Python 3.10 · pandas 1.5 · numpy 1.23 · scikit-learn 1.2 · imbalanced-learn · shap · matplotlib · seaborn.

## 9) Citas y marco de clase
- Clase 4: Regresión lineal/logística · Clase 5: KNN/Árboles · Clase 6: SVM/SGD · Clase 8: Clustering.  
  Material y prácticas de la Tecnicatura.

---

## 10) Bitácora del proceso del proyecto
Este proyecto no nació “ordenado”: errores de rutas y carpetas llevaron a crear `figs/` y `results/`, usar rutas relativas y versionar salidas.  
En modelado, todo quedó dentro de **Pipeline**, se priorizó **recall** y se ajustó el **umbral** por **PR** (aceptando más FP para detección temprana).  
En GitHub aparecieron *mixed line endings*, PDFs tratados como texto y figuras que se ven sólo en **github.dev**; mientras se estabiliza, los **PDF/figuras** se respaldan en **Drive** (enlaces arriba).  
Cada tropiezo dejó una mejora: carpetas prolijas, `.gitattributes`, umbral justificado y resultados reproducibles.

---

© 2025 Ana María Fernández — Tecnicatura en Ciencia de Datos e IA
