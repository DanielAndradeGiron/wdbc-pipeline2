# WDBC Pipeline — validación anidada repetida resistente a fugas

Pipeline profesional y reproducible para clasificar tumores de mama (WDBC, *Wisconsin Diagnostic
Breast Cancer*; clase positiva **M** = maligno). Reorganiza el notebook original
(`notebooks/legacy/`) como un paquete Python ejecutable desde **VS Code** o la terminal, conservando
toda su metodología y corrigiendo sus debilidades (ver **[docs/AUDIT.md](docs/AUDIT.md)**).

```
datos ─► split 80/20 + holdout BLOQUEADO
          │
          ▼  (solo desarrollo)
     CV anidada repetida 10×5 ── 10 candidatos · RandomizedSearchCV interna · sobremuestreo y
          │                       calibración sigmoide dentro de cada fold
          ▼
     selección por tolerancias prácticas ─► Friedman (exploratorio) + Nadeau–Bengio + Holm
          ▼
     umbral OOF totalmente anidado (sens ≥ 0.95, máx. especificidad)
          ▼
     ajuste final + CONGELACIÓN (modelo, hiperparámetros, umbral, checksum)
          ▼
     ÚNICA apertura del holdout ─► IC bootstrap/Clopper–Pearson · sensibilidad al desarrollo
          ▼                         ROC/PR · calibración · DCA · permutación/SHAP/LIME
     puertas de calidad ─► informe .md + .docx · manifiesto SHA-256 · .zip
```

## 1. Puesta en marcha en VS Code

Requisitos: **Python 3.10–3.12** (recomendado 3.12; algunas ruedas de CatBoost/SHAP pueden no existir
aún para versiones más nuevas) y VS Code con las extensiones recomendadas (VS Code las ofrece al abrir
la carpeta).

1. `Archivo → Abrir carpeta…` → seleccione `wdbc-pipeline/`.
2. `Ctrl+Shift+P → Tasks: Run Task → Setup: crear .venv e instalar (dev + explain)`
   *(o en la terminal)*:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
   python -m pip install -U pip
   pip install -e ".[all]"
   ```
3. `Ctrl+Shift+P → Python: Select Interpreter` → `.venv`.
4. Verifique: `python -m wdbc_pipeline list-models` (todos deben decir `OK`).
5. Ejecute:
   - **F5 → "Pipeline: quick"** — prueba estructural (~2 min). *No reporte sus números.*
   - **F5 → "Pipeline: completo"** — análisis publicable (minutos a horas según núcleos).
   - Pestaña **Testing** (matraz) → ejecutar las pruebas.

## 2. Uso por línea de comandos

```bash
python -m wdbc_pipeline run --config configs/quick.yaml            # prueba rápida
python -m wdbc_pipeline run --config configs/default.yaml          # análisis completo
python -m wdbc_pipeline run --config configs/notebook_replica.yaml # réplica fiel del notebook

# Etapas sueltas sobre una ejecución existente (p. ej. rehacer figuras e informe):
python -m wdbc_pipeline run -c configs/default.yaml \
    --run-dir outputs/runs/full_<hash> --stages curves explain report

# Sobrescribir parámetros sin editar el YAML (cambian el hash → nuevo run):
python -m wdbc_pipeline run -c configs/default.yaml --set cv.outer_repeats=10 --set runtime.n_jobs=6

python -m wdbc_pipeline show-config -c configs/default.yaml   # configuración resuelta + hash
```

Etapas (orden canónico): `data benchmark select stats threshold final holdout uncertainty curves explain report`.
Si el proceso se interrumpe durante `benchmark`, vuelva a lanzar el mismo comando: los bloques ya
calculados se cargan desde `checkpoints/`.

También hay atajos `make quick | run | replica | test | lint` (Linux/macOS).

## 3. Configuraciones

| Archivo | Uso |
|---|---|
| `configs/default.yaml` | Análisis completo con las mejoras de la auditoría (fuente `sklearn`, sin doble balanceo, submuestreo sin reemplazo). |
| `configs/quick.yaml` | Hereda de *default*; CV 3×1 y bootstrap reducidos. Solo para comprobar que todo funciona. |
| `configs/notebook_replica.yaml` | Reproduce las decisiones originales (URL de Google Sheets, doble balanceo, bootstrap). Use con `requirements-notebook-pins.txt`. |

Fuentes de datos (`data.source`): `sklearn` (copia oficial de WDBC, sin internet), `csv`
(`data.csv_path`) o `url`. Fije `data.expected_sha256` para que cualquier cambio en los datos detenga
la ejecución.

## 4. Garantías de integridad

- **Hash de configuración**: cada run vive en `outputs/runs/<run_name>_<hash>/`; no se pueden mezclar
  resultados de configuraciones distintas en el mismo directorio.
- **Holdout bloqueado por código**: solo se abre tras congelar modelo y umbral, se verifica su SHA-256,
  una sola vez; reabrir exige `--force-holdout` y queda en `metadata/holdout_access_log.jsonl`.
- **Puertas de calidad** (`metadata/quality_checks.json`): split disjunto y completo, probabilidades
  válidas, una predicción OOF por registro, consistencia del modelo final, arquitectura del stacking, IC
  disponibles. Si alguna falla, el informe no se genera.
- **Manifiesto SHA-256** de todos los artefactos (`artifact_checksum_manifest.csv`).

## 5. Estructura

```
wdbc-pipeline/
├── configs/                  # YAML (default, quick, notebook_replica)
├── src/wdbc_pipeline/
│   ├── cli.py, __main__.py   # interfaz de línea de comandos
│   ├── pipeline.py           # orquestación por etapas
│   ├── config.py             # dataclasses inmutables, herencia, validación, hash
│   ├── artifacts.py          # RunContext, tablas/figuras/JSON, checksums
│   ├── data.py               # adquisición, auditoría, split, HoldoutGuard
│   ├── sampling.py           # sobremuestreo fold-local (meta-estimador sklearn)
│   ├── models.py             # registro de 10 candidatos y arquitecturas
│   ├── tuning.py             # búsqueda interna + calibración
│   ├── benchmark.py          # CV anidada repetida paralela con checkpoints
│   ├── selection.py          # resumen y cascada de tolerancias
│   ├── stats.py              # Friedman, Nadeau–Bengio, Holm
│   ├── thresholds.py         # umbral OOF anidado
│   ├── final.py              # ajuste final y congelación
│   ├── evaluation.py         # holdout, IC, sensibilidad, ROC/PR, calibración, DCA
│   ├── explain.py            # permutación, SHAP, LIME con control de fidelidad
│   ├── plotting.py           # figuras de estilo editorial
│   ├── report.py             # puertas de calidad, informes MD/DOCX, zip
│   └── logging_utils.py
├── tests/                    # 40 pruebas (unitarias + end-to-end marcadas "slow")
├── docs/AUDIT.md             # evaluación de calidad del notebook original
├── notebooks/legacy/         # notebook original (trazabilidad)
├── .vscode/                  # launch, tasks, settings, extensiones
├── .github/workflows/ci.yml  # lint + tipos + tests + run rápido
├── pyproject.toml · requirements*.txt · Makefile · .pre-commit-config.yaml
```

## 6. Salidas de una ejecución

```
outputs/runs/<run>_<hash>/
├── TECHNICAL_RESULTS_REPORT.md · PUBLICATION_RESULTS_REPORT.docx · README.md
├── tables/      # CSV canónicos (+ Excel con formato)
├── figures/     # PNG/PDF/SVG (600 dpi en el modo completo)
├── models/      # modelo calibrado congelado (.joblib)
├── metadata/    # config resuelta, hash, versiones, manifiestos, decisiones congeladas, puertas
├── data/        # snapshot, desarrollo y holdout bloqueado
├── logs/        # pipeline.log completo
└── artifact_checksum_manifest.csv
```

## 7. Calidad del código

```bash
pytest -m "not slow"   # rápidas (~5 s)
pytest                 # incluye end-to-end (~1 min)
ruff check src tests && ruff format --check src tests
mypy src
```

## 8. Límites de interpretación

El holdout es **interno**: los resultados no demuestran transportabilidad externa, utilidad clínica
prospectiva ni efectos causales. El umbral secundario es provisional hasta validación externa; la DCA y
la explicabilidad son *post hoc*.
