# Auditoría de calidad del pipeline original

**Objeto auditado:** `notebooks/legacy/pipeline_original_2026-08-06.ipynb` (19 celdas, Google Colab)
**Alcance:** rigor metodológico, reproducibilidad, ingeniería de software y operación.
**Resultado del notebook:** LightGBM seleccionado; holdout interno (n=114) AUC 0.991, MCC 0.944 @0.50.

---

## 1. Veredicto

| Dimensión | Nota (1–5) | Comentario breve |
|---|:---:|---|
| Diseño estadístico / validación | **4.5** | Muy por encima de lo habitual: CV anidada repetida, holdout bloqueado, umbral anidado, t corregida + Holm. |
| Prevención de fuga de información | **4.5** | Preprocesado y sobremuestreo dentro de cada fold; stacking con base learners autocontenidos. |
| Reproducibilidad | **2.5** | Semillas y manifiestos, pero datos de una URL mutable sin verificación y entorno solo-Colab. |
| Ingeniería de software | **1.5** | Monolito de celdas, código denso de una línea, estado global, sin tests, sin logging, sin tipos. |
| Operación / rendimiento | **2.0** | ~3.5 h en serie, sin checkpoints; una interrupción obliga a repetir todo. |
| Gobernanza del holdout | **3.0** | Bloqueo "por convención" (comentario), no impuesto por código. |

**Conclusión:** la *ciencia* del pipeline es sólida; la *ingeniería* no estaba a la altura. El proyecto
reorganizado conserva íntegramente la metodología y corrige los puntos débiles listados abajo.

---

## 2. Fortalezas que se conservan

1. **Holdout estratificado inmutable** (80/20, semilla 42) separado antes de cualquier análisis.
2. **Preprocesado y sobremuestreo fold-local**: nunca se ajustan con filas de validación.
3. **Stacking sin fuga**: cada base learner tiene su propio `prep → sampler → clf`; meta-features por CV interna.
4. **Calibración sigmoide de todos los candidatos** dentro de cada fold externo (comparación justa de Brier/ECE).
5. **Umbral operativo totalmente anidado** (OOF) para el algoritmo ya congelado.
6. **Estadística consciente de la dependencia**: t de Nadeau–Bengio + Holm; Friedman declarado exploratorio.
7. **Incertidumbre** por bootstrap estratificado + IC exactos de Clopper–Pearson.
8. **LIME condicionado a fidelidad** (R² ≥ 0.50) — se suprimió correctamente (R² = 0.078).
9. **Manifiestos** de split, configuración congelada y checksums de artefactos.

---

## 3. Hallazgos y correcciones

Severidad: 🔴 alta · 🟠 media · 🟡 baja

| # | Sev. | Hallazgo | Riesgo | Corrección en el proyecto |
|---|:---:|---|---|---|
| H1 | 🔴 | Datos leídos de una **URL de Google Sheets mutable**; el SHA-256 se registraba pero **nunca se verificaba**. | Re-ejecuciones con datos distintos sin aviso. | `data.expected_sha256`: si el hash no coincide, la ejecución se detiene. Fuente `sklearn` (WDBC oficial, offline) por defecto; `csv` local o `url` opcionales. Snapshot + hash guardados en cada run. |
| H2 | 🔴 | Bloqueo del holdout **solo por convención** (comentario en la celda 4). | Reapertura accidental → holdout convertido en conjunto de ajuste. | `HoldoutGuard`: exige configuración congelada, verifica SHA-256 del archivo, **permite una sola apertura**; reabrir requiere `--force-holdout` y queda registrado en `holdout_access_log.jsonl`. La puerta de calidad informa el nº de aperturas. |
| H3 | 🟠 | **Doble balanceo** en RF/ExtraTrees (y en Voting/Stacking): `class_weight="balanced"` **y** `RandomOverSampler`. | Sobrecorrección de la clase minoritaria, probabilidades sesgadas antes de calibrar; comparación desigual con otros modelos. | Por defecto solo sobremuestreo (`class_weight_with_oversampling: false`). La réplica fiel (`notebook_replica.yaml`) lo restaura. |
| H4 | 🟠 | Sensibilidad de desarrollo con **bootstrap con reemplazo** antes de `CalibratedClassifierCV(cv=3)`. | Copias de la misma fila caen en el fold de ajuste y en el de calibración → calibración optimista. | Modo `subsample` (80 % estratificado **sin** reemplazo) por defecto; `bootstrap` disponible para réplica. |
| H5 | 🟠 | **Ejecución en serie** (`n_jobs=1`) ≈ 12 562 s y **sin checkpoints**. | 3.5 h por intento; cualquier corte obliga a empezar de cero. | Bloques externos en paralelo con `joblib` (estimadores internos `n_jobs=1` para evitar sobresuscripción); checkpoint por bloque y reanudación automática ligada al hash de configuración. |
| H6 | 🟠 | **Entorno solo-Colab** (`%pip` en celda, `google.colab.files`), sin `pyproject`/lockfile. | No ejecutable fuera de Colab; dependencias frágiles (imblearn suele ir por detrás de scikit-learn). | Paquete instalable (`pyproject.toml`), `requirements*.txt`, pines originales conservados, CI. Sobremuestreo implementado como meta-estimador nativo de scikit-learn (`RandomOverSampledClassifier`) → se elimina `imbalanced-learn`; Holm implementado → se eliminan `statsmodels` y `scikit-posthocs`. |
| H7 | 🟠 | Código monolítico: estado global entre celdas, líneas con 5–10 sentencias, `print` en lugar de logging, sin tipos ni tests. | Errores difíciles de detectar; imposible revisar o reutilizar. | Paquete `src/wdbc_pipeline` con módulos de responsabilidad única, tipado (mypy limpio), `logging` a consola y archivo, ruff, **40 tests** (unitarios + end-to-end). |
| H8 | 🟡 | `ece()` devolvía **0** cuando todas las probabilidades eran idénticas (bordes de cuantiles colapsados). | ECE perfecto falso en modelos degenerados. | Se trata como un único bin: `|ȳ − p̄|`. Cubierto por test. |
| H9 | 🟡 | La curva de calibración usaba **10 bins fijos** en código, independiente de `ece_bins`. | Incoherencia entre figura y métrica. | Ambas usan `evaluation.ece_bins`; la curva se exporta también como tabla. |
| H10 | 🟡 | `VotingSoft` y `Stacking` **no se afinan** (rejilla vacía) mientras el resto sí. | Comparación asimétrica (desfavorece a los ensambles). | Se mantiene por fidelidad, pero se documenta y las rejillas son configurables en YAML. |
| H11 | 🟡 | LightGBM se eligió por **desempate de tiempo** frente a CatBoost (mayor F1 medio) sin explicitarlo en el informe. | Revisores pueden interpretarlo como selección *post hoc*. | `selection_rule.json` y el informe incluyen una nota automática cuando el ganador ≠ mejor F1 medio, citando las tolerancias preespecificadas. |
| H12 | 🟡 | Friedman sobre bloques de CV repetida **no independientes**. | p-valor anticonservador. | Se mantiene etiquetado como exploratorio; la inferencia principal sigue siendo Nadeau–Bengio + Holm. |
| H13 | 🟡 | Exportación a Excel con formato **celda a celda** (lenta en tablas de miles de filas). | Minutos perdidos en tablas de predicciones. | Anchos calculados vectorialmente; tablas > 50 000 filas solo en CSV (canónico). |
| H14 | 🟡 | Umbral OOF (0.14) trasladado a un modelo reajustado sobre todo el desarrollo. | El umbral puede no transferirse exactamente. | Se mantiene el diseño (es correcto) y el informe lo declara provisional hasta validación externa. |
| H15 | 🟡 | Seeds dependientes de la posición del modelo en la lista. | Cambiar la lista cambia los resultados de otros modelos. | Semillas ligadas al índice en el registro fijo (`SEED + 1000·k + j`), estables aunque se omitan modelos. |

---

## 4. Equivalencias notebook → proyecto

| Celda | Contenido | Módulo |
|---|---|---|
| 0–1 | dependencias, imports | `pyproject.toml`, `requirements*.txt` |
| 2 | `Config` | `configs/*.yaml` + `config.py` (validación, herencia, hash) |
| 3 | exportación, estilo, versiones | `artifacts.py`, `plotting.py` |
| 4 | datos, split, bloqueo | `data.py` (`HoldoutGuard`) |
| 5 | arquitecturas | `models.py`, `sampling.py` |
| 6 | métricas, tuning, selección | `metrics.py`, `tuning.py`, `selection.py` |
| 7 | CV anidada repetida | `benchmark.py` |
| 8 | ranking | `selection.py` |
| 9 | Friedman, Nadeau–Bengio, Holm | `stats.py` |
| 10 | umbral OOF anidado | `thresholds.py` |
| 11 | ajuste final | `final.py` |
| 13–16 | holdout, IC, sensibilidad, curvas, DCA | `evaluation.py` |
| 17 | SHAP / LIME | `explain.py` (+ importancia por permutación sin dependencias) |
| 18 | puertas de calidad, informes, zip | `report.py` |

---

## 5. Recomendaciones pendientes (fuera del alcance del código)

1. **Validación externa** (otra cohorte/centro) antes de cualquier afirmación clínica.
2. Considerar **afinar los ensambles** o compararlos contra versiones sin afinar de los modelos simples.
3. Si se publica la réplica del notebook, **fijar `expected_sha256`** tras la primera descarga.
4. Reportar el **número de aperturas del holdout** (debe ser 1) en el manuscrito.
