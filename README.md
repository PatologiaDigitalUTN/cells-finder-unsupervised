# Clustering y Segmentación de Núcleos con BiomedCLIP

Pipeline de clustering jerárquico multi-paso para segmentación automática de núcleos en imágenes citológicas usando embeddings de **BiomedCLIP**.

## ⚡ Quick Start

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline
python main.py

# O ver tutorial paso a paso
python example.py
```

## 🎯 Descripción

Pipeline automático de 4 pasos + limpieza:
1. **Fondo vs Tejido** (KMeans k=2)
2. **Núcleos vs Citoplasma** (KMeans k=2)
3. **Núcleo en Citoplasma vs Suelto** (KMeans k=2)
4. **Refinamiento de Núcleos** (KMeans k=2)
5. **Limpieza** por componentes conexas
6. **Agrupación** en núcleos finales
7. **Evaluación** contra ground truth

## Estructura del Proyecto

```
.
├── main.py                          # Script principal
├── requirements.txt                 # Dependencias
├── README.md                        # Este archivo
└── utils/
    ├── __init__.py
    ├── image_processing.py          # Cargar imágenes y boxes
    ├── embeddings.py                # Extracción de embeddings BiomedCLIP
    ├── multi_step_clustering.py     # Lógica de clustering multi-paso
    ├── evaluation.py                # Métricas y evaluación
    └── visualization.py             # Funciones de plotting
```

## 📁 Estructura del Proyecto

```
clustering-segmentacion-biomedclip/
├── main.py                  # Script principal 📌
├── config.py                # Configuración centralizada ⚙️
├── example.py               # Tutorial paso a paso 📚
├── README.md                # Este archivo
├── REFACTOR_NOTES.md        # Notas de refactorización
├── requirements.txt         # Dependencias 📦
│
└── utils/                   # Módulos reutilizables 🔧
    ├── image_processing.py  # Cargar imágenes y boxes
    ├── embeddings.py        # Extraer embeddings
    ├── multi_step_clustering.py  # Pipeline de clustering
    ├── evaluation.py        # Métricas y evaluación
    └── visualization.py     # Plotting y visualización
```

## 🚀 Uso

### Opción 1: Automático (recomendado para Kaggle)

```bash
python main.py
```

Procesa todas las imágenes configuradas y genera:
- Visualizaciones PNG para cada paso
- JSON con métricas (TP, FP, precision, recall, F1)
- Resumen global de resultados

### Opción 2: Tutorial (aprender paso a paso)

```bash
python example.py
```

Procesa una imagen mostrando cada paso con visualización interactiva.

### Opción 3: Personalizado en Jupyter

```python
from config import KaggleConfig
from main import run_pipeline

cfg = KaggleConfig()
cfg.images_to_analyze = [150, 151, 152]
cfg.out_root = './mi_output'

metrics = run_pipeline(cfg)
```

### Opción 4: Reutilizar módulos

```python
from utils.embeddings import get_all_patch_embeddings_from_image
from utils.multi_step_clustering import run_block_clustering_on_embeddings

# Usar en tu proyecto
embeddings = get_all_patch_embeddings_from_image(img, model, preprocess)
clusters = run_block_clustering_on_embeddings(embeddings, n_clusters=2)
```

## ⚙️ Configuración

Editar `config.py` para personalizar:

```python
# Rutas de datos
json_path = '/kaggle/input/cric-dataset/classifications.json'
base_path = '/kaggle/input/cric-dataset'

# Parámetros de procesamiento
box_size = 224                    # Tamaño de tile para BiomedCLIP
boxes_size_gt = 60                # Tamaño de box GT
min_patches_componente = 3        # Mínimo de patches por componente
dilate_px_componentes = 32        # Dilatación para cerrar gaps

# Evaluación
match_mode = 'cover_gt'           # Modo de matching
cover_gt_thr = 0.20              # Umbral de cobertura

# Output
out_root = './eval_multistep'    # Directorio de resultados
```

O crear una configuración personalizada:

```python
from config import BaseConfig

class MiConfig(BaseConfig):
    box_size = 512
    min_patches_componente = 5
    out_root = './mi_output'
```

## 📊 Output

Para cada imagen procesada:

```
{idx:04d}_{nombre}/
├── 01_fondo_vs_tejido.png          # Paso 1: Fondo vs Tejido
├── 02_nucleos_vs_cyto.png          # Paso 2: Núcleos vs Citoplasma
├── 03_nucleo_cito_vs_suelto.png    # Paso 3: Núcleo en Citoplasma vs Suelto
├── 04_refinamiento_final.png       # Paso 4: Refinamiento
├── 05_limpieza.png                 # Resultado tras limpieza
├── 06_grupos_vs_GT.png             # Resultado final vs GT
└── metrics.json                    # Métricas (TP, FP, P, R, F1)
```

Resumen global:
```
metrics_summary.json               # Resultado de todas las imágenes
```

## 📚 API Principal

### `main.py`
```python
from main import run_pipeline
from config import KaggleConfig

cfg = KaggleConfig()
metrics = run_pipeline(cfg)  # Procesa todas las imágenes
```

### `utils.multi_step_clustering`
```python
from utils.multi_step_clustering import (
    run_block_clustering_on_embeddings,  # KMeans/Agglomerative
    refinar_cluster_con_kmeans,          # Refinar cluster
    decidir_fondo_vs_tejido,             # Auto-decisión paso 1
    decidir_nucleos_vs_citoplasma        # Auto-decisión paso 2
)
```

### `utils.evaluation`
```python
from utils.evaluation import (
    evaluar_grupos_vs_boxes_plus,        # Métricas
    limpiar_patches_por_componentes_mask,# Limpieza
    agrupar_patches_en_grupos            # Agrupación
)
```

### `utils.visualization`
```python
from utils.visualization import (
    visualizar_clusters_basicos,
    visualizar_limpieza_patches,
    visualizar_grupos_vs_boxes
)
```

## 📦 Requisitos

```
torch>=1.9.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
open-clip-torch>=2.23.0
transformers>=4.35.2
```

## 🔍 Troubleshooting

**Error: "No se puede leer imagen"**
- Verificar rutas en `config.py`
- Verificar índice en `images_to_analyze`

**Error: "Muy pocos puntos para clustering"**
- Aumentar `box_size`
- Verificar tamaño de imagen

**Error al cargar BiomedCLIP**
- Necesita conexión a internet (primera vez)
- Asegurar `open-clip-torch` instalado

## 📖 Ver También

- `REFACTOR_NOTES.md` - Detalles de la refactorización
- `example.py` - Tutorial paso a paso
- `README.md` - Este archivo
