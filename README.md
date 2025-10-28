# Predicción de situaciones de trata de personas (2020–2024)

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

📄 **Informe completo (PDF):** [Ver](https://drive.google.com/file/d/1KOsoBH0DmTL9VTpilG6K0Wa1Wn1yGD0I/view?usp=drive_link) · [Descargar el informe (PDF)](https://drive.google.com/uc?export=download&id=1KOsoBH0DmTL9VTpilG6K0Wa1Wn1yGD0I)
📁 **Carpeta del proyecto (Drive):** [Abrir](https://drive.google.com/drive/folders/1Pi_5rFwRCzmmJpSQl1gV6k_Ke6B7OvzF?usp=drive_link)

---

## 2) Datos
### Fuente y alcance del dataset
- **Origen**: Oficina de Rescate y Acompañamiento (AR), Argentina.
- **Cobertura temporal**: ene-2020 a dic-2024 (corte incluido).
- **Fecha de adquisición**: dd/mm/aaaa.
- **Uso y estado**: Con fines educativos; datos normalizados y **anonimizados** (sin PII).
- **Accesos**:
  - CSV procesado (canónico) — *si el repo no lo puede alojar por tamaño/privacidad*: [Descargar desde Drive](https://drive.google.com/file/d/1_ttFnO6qTHTiBnFyFbd5pzR4SB0cjPgc/view?usp=sharing)
  - Estructura de carpetas del proyecto en este repositorio (`data/raw`, `data/processed`, `results`, `figs`).

### Descripción del dataset (resumen)
- **Instancias**: 7.848
- **Características**: 26 columnas (ver diccionario)
- **Target**: `es_trata` (1/0) — balance: 1=54% (4241), 0=46% (3607)
- **Tipos y nulos**: ver `results/nulos_antes.csv` y `results/nulos_despues.csv`  
  (incluyen conteos, % nulos y dtype por columna).
- **Transformaciones principales**:
  - Normalización de strings y categorías (provincias/localidades/nacionalidad).
  - Derivación temporal (`anio`, `mes`, `trimestre`, `dia_semana`, `es_fin_semana`, `mes_sin`, `mes_cos`).
  - Construcción robusta de `es_trata` (reglas auditadas).
  - Eliminación de `hora_ingreso` por estar vacía.

### Diccionario de datos (extracto)
| columna                      | tipo     | descripción breve                              | % nulos | # únicos |
|-----------------------------|----------|-----------------------------------------------|---------|---------|
| `es_trata`                  | int (0/1)| Etiqueta binaria (objetivo)                   | 0%      | 2       |
| `consultante_provincia`     | string   | Provincia normalizada (INDEC)                 | ~40%    | ~25     |
| `consultante_localidad`     | string   | Localidad normalizada                         | ~81%    | ~351    |
| `consultante_nacionalidad`  | string   | Nacionalidad normalizada                      | ~76%    | ~18     |
| `consultante_edad_aparente` | float    | Edad aparente                                 | ~15%    | ~88     |
| `anio` `mes` `trimestre`    | int      | Derivadas temporales                          | —       | —       |
| `mes_sin` `mes_cos`         | float    | Componentes cíclicos del mes                  | —       | —       |
> El detalle completo está en `results/nulos_*.csv`.

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

## 7) Consideraciones éticas y privacidad
- Anonimización estricta, no publicar PII. Uso educativo con fines de mejora operativa.

## 8) Entorno
- Python 3.10 · pandas 1.5 · numpy 1.23 · scikit-learn 1.2 · imbalanced-learn · shap · seaborn

## 9) Citas y marco de clase
- Clase 4: Regresión lineal y logística; Clase 5: KNN y Árboles; Clase 6: SVM/SGD; Clase 8: Clustering (material teórico y prácticas).

## 10) Bitácora del proceso del proyecto

Este proyecto no nació “ordenado”. Empezó como casi todos: con ilusión, con prisa y con algunas torpezas que, con el tiempo, se volvieron aprendizaje. La primera piedra fue tan simple como cruel: las rutas. Guardaba gráficos y tablas como si las carpetas existieran por arte de magia; no existían. Ese primer FileNotFoundError me obligó a crear figs/ y results/, a adoptar rutas relativas y a entender que la prolijidad del directorio también es ciencia de datos.

Después vino la fase de modelado. Ahí aprendí que un buen pipeline es más que un modelo que “anda”. Encerré imputación, codificación y escalado dentro del Pipeline, hice CV estratificada y prioricé recall. La decisión de ajustar el umbral 0.328 no fue capricho: comparé contra 0.5, miré PR y acepté más FP para ganar detección temprana. Hubo celdas rebeldes (la 12, la 16, la 21…) que mostraron que renombrar una variable en una celda rompe tres más adelante. Tomé nota: nombres estables, comentarios claros y resultados versionados en results/.

La parte GitHub fue otra novela. El editor web me tiró el aviso de “mixed line endings” y terminé editando en github.dev (la tecla .) para forzar LF. Cuando subí el PDF, el visor de GitHub no lo pudo abrir: por orden de reglas en .gitattributes, el PDF se trató como texto y quedó inutilizable. Corregí eso con -text para binarios y renormalicé el repo… pero, aun así, hoy el informe no se visualiza en GitHub. Por ese motivo, lo voy a publicar como enlace a Google Drive dentro del README, hasta estabilizar la visualización en el repo.

Con los gráficos pasó algo parecido: en la vista normal de GitHub no se ven (solo puedo verlos entrando a GitHubDev). Mientras persista esa limitación, el README no puede mostrarlos embebidos; voy a dejar las referencias a los archivos en figs/ y, cuando corresponda, los enlaces a Drive como respaldo.

Al final, la seguidilla de problemas terminó siendo un plan de estudios paralelo: ordenar carpetas, blindar el preprocesamiento, justificar el umbral por objetivo operativo, cuidar los finales de línea y documentar limitaciones técnicas (PDF por Drive y figuras visibles solo en GitHubDev por ahora). Si este README se ve bien, si las métricas reproducen el 0.328 elegido y si el flujo está claro, es porque cada tropiezo dejó una marca: una carpeta creada a tiempo, una regla en .gitattributes, un enlace a Drive, una nota sobre visualización. Así, de a poco, el proyecto empezó a parecerse al proyecto que quería hacer.
---

© 2025 Ana María Fernández — Tecnicatura en Ciencia de Datos e IA
