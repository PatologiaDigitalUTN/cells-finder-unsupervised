# Estrategias de Evaluación

## Descripción

El sistema ahora soporta **dos estrategias de evaluación** para calcular las métricas de rendimiento del clustering de núcleos:

1. **Estrategia de Segmentación** (`segmentation`) - **RECOMENDADA**
2. **Estrategia de Bounding Boxes** (`bbox`) - Legacy

---

## 🎯 Estrategia de Segmentación (Recomendada)

### Características

- ✅ Usa las **máscaras de segmentación** directamente del dataset COCO
- ✅ Calcula métricas **pixel-wise** (pixel por pixel)
- ✅ Métricas precisas: **DICE**, **IoU**, **Precision**, **Recall**, **F1**
- ✅ No requiere agrupamiento de patches
- ✅ Visualización con overlays de TP/FP/FN en colores

### Métricas Calculadas

```python
{
    'dice': float,              # Coeficiente DICE (2*TP / (2*TP + FP + FN))
    'iou': float,               # Intersection over Union
    'precision': float,         # TP / (TP + FP) - pixel-wise
    'recall': float,            # TP / (TP + FN) - pixel-wise
    'f1_coverage': float,       # F1 score - pixel-wise
    'tp_pixels': int,           # Píxeles verdaderos positivos
    'fp_pixels': int,           # Píxeles falsos positivos
    'fn_pixels': int,           # Píxeles falsos negativos
    'tn_pixels': int            # Píxeles verdaderos negativos
}
```

### ¿Cómo funciona?

1. **Carga las máscaras GT**: Lee los polígonos de segmentación del COCO JSON
2. **Convierte a binario**: Transforma polígonos a máscaras binarias (0 o 1)
3. **Crea máscara de predicción**: Convierte patches predichos a máscara binaria
4. **Calcula métricas pixel-wise**: Compara píxel a píxel las máscaras

### Visualización

- **Verde**: Verdaderos Positivos (TP) - Píxeles correctamente detectados
- **Rojo**: Falsos Positivos (FP) - Píxeles incorrectamente detectados
- **Azul**: Falsos Negativos (FN) - Píxeles de núcleo no detectados
- **Contorno amarillo**: Ground truth (máscaras reales)

---

## 📦 Estrategia de Bounding Boxes (Legacy)

### Características

- ⚠️ Método anterior preservado por compatibilidad
- ⚠️ Agrupa patches en componentes conexas
- ⚠️ Calcula bounding boxes envolventes
- ⚠️ Evalúa overlap con GT boxes
- ⚠️ Menos preciso que segmentación

### Métricas Calculadas

```python
{
    'groups_TP': int,              # Grupos correctamente detectados
    'groups_FP': int,              # Grupos falsos positivos
    'group_precision': float,      # TP / (TP + FP) - a nivel de grupo
    'gt_recall_coverage': float,   # GT cubiertos / Total GT
    'f1_coverage': float,          # F1 score - a nivel de grupo
    'gt_hit': int,                 # Ground truths cubiertos
    'gt_total': int,               # Total de ground truths
    'pred_hits': list              # Lista booleana de hits por grupo
}
```

### ¿Cómo funciona?

1. **Limpia patches**: Elimina componentes pequeñas con `limpiar_patches_por_componentes_mask()`
2. **Agrupa patches**: Une patches cercanos con `agrupar_patches_en_grupos()`
3. **Calcula bounding boxes**: Obtiene el rectángulo envolvente de cada grupo
4. **Evalúa overlap**: Compara área de overlap con GT boxes

### Visualización

- **Verde**: Grupos verdaderos positivos (TP)
- **Rojo**: Grupos falsos positivos (FP)
- **Rectángulo azul**: Ground truth cubiertos
- **Rectángulo naranja**: Ground truth no cubiertos

---

## 🚀 Uso

### En el Notebook

```python
# CELDA 2: Configuración
EVALUATION_STRATEGY = 'segmentation'  # O 'bbox'

# CELDA 3: Ejecutar pipeline
pipeline = ImageProcessingPipeline(
    json_path=JSON_PATH,
    images_dir=IMAGES_DIR,
    model_name=MODEL_NAME,
    preprocessing_method=PREPROCESSING_METHOD,
    category_ids=CATEGORY_IDS,
    visualize=VISUALIZE,
    n_steps=N_STEPS,
    evaluation_strategy=EVALUATION_STRATEGY  # ⭐ Nueva opción
)
```

### En Scripts Python

```python
from process_image import ImageProcessingPipeline

# Con estrategia de segmentación (recomendado)
pipeline = ImageProcessingPipeline(
    json_path='data.json',
    images_dir='images/',
    evaluation_strategy='segmentation'
)

# Con estrategia de bbox (legacy)
pipeline = ImageProcessingPipeline(
    json_path='data.json',
    images_dir='images/',
    evaluation_strategy='bbox'
)
```

### Generar Reporte HTML

```python
from process_image import generate_html_report

generate_html_report(
    json_path=JSON_PATH,
    images_dir=IMAGES_DIR,
    output_path='reporte.html',
    n_steps=4,
    model_name='biomedclip',
    preprocessing_method='clahe',
    category_ids=[4, 5],
    evaluation_strategy='segmentation'  # ⭐ Nueva opción
)
```

---

## 🧪 Testing

Usa el script `test_strategies.py` para probar ambas estrategias:

```bash
# Probar estrategia de segmentación
python test_strategies.py --mode segmentation

# Probar estrategia de bbox
python test_strategies.py --mode bbox

# Comparar ambas estrategias
python test_strategies.py --mode compare
```

---

## 📊 Comparación de Estrategias

| Característica | Segmentación | BBox |
|---------------|-------------|------|
| **Precisión** | ⭐⭐⭐⭐⭐ Alta | ⭐⭐⭐ Media |
| **Velocidad** | ⭐⭐⭐⭐ Rápida | ⭐⭐⭐ Media |
| **Métricas** | DICE, IoU, pixel-wise | Groups TP/FP |
| **Visualización** | TP/FP/FN overlay | Bounding boxes |
| **Requiere grouping** | ❌ No | ✅ Sí |
| **Usa máscaras GT** | ✅ Sí | ❌ No (solo boxes) |

---

## 🔧 Arquitectura Interna

### Patrón de Diseño: Strategy Pattern

```
├── EvaluationStrategy (abstract base)
│   ├── load_ground_truth()
│   ├── evaluate_patches()
│   └── get_visualization_data()
│
├── SegmentationStrategy
│   ├── Convierte polígonos → máscaras
│   ├── Calcula DICE, IoU
│   └── Retorna TP/FP/FN masks
│
└── BoundingBoxStrategy
    ├── Agrupa patches
    ├── Calcula overlap
    └── Retorna grupos y boxes
```

### Integración con ClusteringTree

```python
# ClusteringTree ahora recibe:
tree = ClusteringTree(
    patch_data,
    img,
    ground_truth,      # No solo boxes, cualquier formato
    H, W,
    evaluation_strategy  # Instancia de la estrategia
)

# Y evalúa usando:
metrics = evaluation_strategy.evaluate_patches(
    patches,
    img,
    ground_truth
)
```

---

## ❓ FAQ

**P: ¿Qué estrategia debo usar?**
R: **Segmentación**. Es más precisa y usa los datos de segmentación disponibles.

**P: ¿Por qué mantener la estrategia bbox?**
R: Para comparación con resultados previos y reproducibilidad.

**P: ¿Puedo agregar más estrategias?**
R: Sí, crea una clase que herede de `EvaluationStrategy` en `utils/evaluation_strategies.py`.

**P: ¿Cambian las métricas al cambiar de estrategia?**
R: Sí, pero ambas calculan `f1_coverage` para compatibilidad.

---

## 📝 Referencias

- **DICE Coefficient**: Sørensen–Dice coefficient
- **IoU**: Intersection over Union (Jaccard index)
- **COCO Format**: [COCO Dataset Format](https://cocodataset.org/#format-data)
- **Polygon Segmentation**: RLE vs Polygon en COCO

---

## 👥 Contribuciones

Para agregar una nueva estrategia de evaluación:

1. Crea una clase en `utils/evaluation_strategies.py`
2. Hereda de `EvaluationStrategy`
3. Implementa los 3 métodos abstractos:
   - `load_ground_truth()`
   - `evaluate_patches()`
   - `get_visualization_data()`
4. Registra en la factory `create_evaluation_strategy()`

---

**Última actualización**: 2025
**Autor**: Marco
**Proyecto**: Cells Finder Unsupervised
