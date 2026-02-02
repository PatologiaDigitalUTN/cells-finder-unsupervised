# 🔥 Guía: Ejecutar en Kaggle Notebooks

## 🚀 Quick Start (5 minutos)

### Paso 1: Crear Kaggle Notebook

1. Ve a [kaggle.com/notebooks](https://www.kaggle.com/notebooks)
2. Click en "New Notebook"
3. Selecciona Python como lenguaje
4. Habilita Internet en Settings (⚙️)

### Paso 2: Agregar Dataset

En Settings (⚙️):
1. Search: "cric-dataset" por martingra
2. Add Dataset
3. Save Notebook

### Paso 3: Copiar Archivos

En tu Notebook, crea estas **celdas** en orden:

---

## 📋 CELDA 1: Setup (Run Once)

```python
# Copiar y pegar en la primera celda
%run kaggle_setup.py
```

Esto:
- ✅ Instala dependencias necesarias
- ✅ Descarga dataset automáticamente
- ✅ Importa todos los módulos

**Esperado:** Demora 2-3 minutos (primera vez)

---

## 📋 CELDA 2: Opción A - Pipeline Automático

```python
# Procesar todas las imágenes
from config import KaggleConfig
from main import run_pipeline

cfg = KaggleConfig()
cfg.images_to_analyze = [150, 151, 152]  # Modificar según necesites
cfg.out_root = '/kaggle/working/resultados'

metrics = run_pipeline(cfg)

# Ver resumen
import json
print("\n📊 RESUMEN DE MÉTRICAS:")
for m in metrics:
    print(f"  {m['fname']}: F1={m['f1_coverage']:.3f}")
```

---

## 📋 CELDA 3: Opción B - Tutorial Paso a Paso

```python
# Ver tutorial completo (procesa una imagen)
# Esto es el contenido de example.py adaptado para Kaggle

from open_clip import create_model_from_pretrained, get_tokenizer
from utils.image_processing import load_image_and_boxes_from_json_cropped
from utils.embeddings import get_all_patch_embeddings_from_image
from utils.multi_step_clustering import (
    run_block_clustering_on_embeddings,
    refinar_cluster_con_kmeans,
    decidir_fondo_vs_tejido,
    decidir_nucleos_vs_citoplasma
)
from utils.evaluation import agrupar_patches_en_grupos, evaluar_grupos_vs_boxes_plus
from utils.visualization import visualizar_clusters_basicos

# Cargar modelo
print("Cargando BiomedCLIP...")
model, preprocess = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
print("✅ Modelo cargado\n")

# Cargar imagen
print("Cargando imagen...")
img, boxes, fname = load_image_and_boxes_from_json_cropped(
    '/kaggle/input/cric-dataset/classifications.json',
    '/kaggle/input/cric-dataset',
    index=150,  # Cambiar índice aquí
    boxes_size=60,
    block_size=224
)
H, W = img.shape[:2]
print(f"✅ {fname} ({W}x{H})\n")

# Extraer embeddings
print("Extrayendo embeddings...")
patch_data = get_all_patch_embeddings_from_image(
    img, model, preprocess, tile_size=224
)
print(f"✅ {len(patch_data)} patches\n")

# PASO 1
print("PASO 1: Fondo vs Tejido")
clustered_k2 = run_block_clustering_on_embeddings(patch_data, n_clusters=2)
fondo_id, tejido_id, _ = decidir_fondo_vs_tejido(img, clustered_k2)
visualizar_clusters_basicos(img, clustered_k2, boxes=boxes, title="Paso 1")

# PASO 2
print("\nPASO 2: Núcleos vs Citoplasma")
paso_2 = refinar_cluster_con_kmeans(
    clustered_k2, cluster_id=tejido_id, nuevo_k=2,
    cluster_field='cluster', new_field='subcluster'
)
nucleos_id, cyto_id, _ = decidir_nucleos_vs_citoplasma(img, paso_2)
visualizar_clusters_basicos(img, paso_2, boxes=boxes, title="Paso 2")

# ... continúa con pasos 3 y 4 si lo deseas
```

---

## 📋 CELDA 4: Opción C - Análisis Personalizado

```python
# Procesar una imagen específica con control total

from open_clip import create_model_from_pretrained, get_tokenizer
from utils.image_processing import load_image_and_boxes_from_json_cropped
from utils.embeddings import get_all_patch_embeddings_from_image
from utils.multi_step_clustering import run_block_clustering_on_embeddings

# Cargar modelo (reutiliza del setup si ya está cargado)
# model, preprocess = create_model_from_pretrained(...)

# Configuración
IDX = 150  # Cambiar imagen
JSON = '/kaggle/input/cric-dataset/classifications.json'
BASE = '/kaggle/input/cric-dataset'

# Cargar
img, boxes, fname = load_image_and_boxes_from_json_cropped(JSON, BASE, index=IDX)

# Procesar
embeddings = get_all_patch_embeddings_from_image(img, model, preprocess)
clusters = run_block_clustering_on_embeddings(embeddings, n_clusters=2)

print(f"Imagen: {fname}")
print(f"Patches: {len(embeddings)}")
print(f"Clusters: {len(set(c.get('cluster') for c in clusters))}")
```

---

## 📊 Ver Resultados

```python
# Ver métricas guardadas
import json
import os

result_dir = '/kaggle/working/resultados'

for folder in sorted(os.listdir(result_dir)):
    metrics_file = os.path.join(result_dir, folder, 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            m = json.load(f)
        print(f"{m['fname']}: TP={m['groups_TP']} FP={m['groups_FP']} F1={m['f1_coverage']:.3f}")
```

---

## 💡 Tips para Kaggle

### 1. **Descargar Resultados**
```python
# Comprimir resultados para descargar
import shutil
shutil.make_archive('/kaggle/working/resultados', 'zip', '/kaggle/working', 'resultados')
print("✅ Descarga resultados.zip desde 'Output'")
```

### 2. **GPU Automática**
Kaggle automáticamente usa GPU si está disponible. Ver en:
```python
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 3. **Memoria**
Si te corre poco RAM:
```python
import gc
gc.collect()  # Liberar memoria

# O procesar imágenes en batches más pequeños
cfg.images_to_analyze = [150, 151]  # Menos imágenes por vez
```

### 4. **Detener Early**
Si algo tarda demasiado, press el botón de STOP en Kaggle o:
```python
# Interrupt kernel
import os
os.system('pkill -f jupyter')
```

---

## 🐛 Troubleshooting en Kaggle

### Error: "No module named 'cv2'"
**Solución:** Ejecutar Celda 1 (kaggle_setup.py) nuevamente

### Error: "Dataset not found"
**Solución:** 
- Verificar que "cric-dataset" está agregado en Settings
- O especificar ruta manual:
```python
JSON = '/kaggle/input/cric-dataset/classifications.json'
```

### Error: "CUDA out of memory"
**Solución:**
```python
import torch
torch.cuda.empty_cache()

# Y procesar menos imágenes por vez
cfg.images_to_analyze = [150, 151]
```

### Notebook se reinicia
**Solución:**
- Kaggle reinicia notebooks después de 1 hora inactivo
- Continúa desde donde quedó (Kaggle guarda todo)

---

## 📈 Rendimiento Esperado

| Recurso | Kaggle | Local |
|---------|--------|-------|
| GPU | Tesla T4 (~16GB) | Variable |
| CPU | 4 cores | Variable |
| RAM | 13GB | Variable |
| Tiempo/imagen | ~1-2 min | Variable |

**Típico:** 20 imágenes en ~30 minutos en Kaggle

---

## 🔄 Workflow Completo

```
1. Create Notebook → Settings → Add Dataset
2. Celda 1: %run kaggle_setup.py
3. Celda 2: from config import KaggleConfig; cfg = KaggleConfig()
4. Celda 3: metrics = run_pipeline(cfg)
5. Celda 4: Download resultados.zip
6. Analizar resultados localmente si lo deseas
```

---

## 📚 Más Ejemplos

### Solo imagen 150
```python
cfg = KaggleConfig()
cfg.images_to_analyze = [150]
metrics = run_pipeline(cfg)
```

### Todas las imágenes (default)
```python
cfg = KaggleConfig()
metrics = run_pipeline(cfg)  # Usa lista default
```

### Custom con parámetros
```python
cfg = KaggleConfig()
cfg.images_to_analyze = [150, 151, 152]
cfg.min_patches_componente = 5  # Más restrictivo
cfg.dilate_px_componentes = 16  # Menos dilatación
cfg.cover_gt_thr = 0.25  # Threshold más alto
metrics = run_pipeline(cfg)
```

---

## ✅ Checklist

- [ ] Notebook creado en Kaggle
- [ ] Dataset "cric-dataset" agregado
- [ ] Internet habilitado en Settings
- [ ] Celda 1 ejecutada (kaggle_setup.py)
- [ ] Celda 2 ejecutada (pipeline o tutorial)
- [ ] Resultados visualizados/descargados

---

¿Necesitas ayuda con algo específico? 🎯
