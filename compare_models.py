"""
Ejemplo: Comparar múltiples modelos fundacionales.

Este script procesa una imagen con diferentes modelos y compara resultados.
"""

import time
import matplotlib.pyplot as plt
from utils.model_factory import create_model, list_available_models
from utils.image_processing import load_image_and_boxes_from_json_cropped
from utils.embeddings import get_all_patch_embeddings_from_image
from utils.multi_step_clustering import run_block_clustering_on_embeddings
from utils.visualization import visualizar_clusters_basicos

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

JSON_PATH = 'C:\\Users\\mngra\\projects\\AI\\Pap\\PAP_DATA\\CRIC\\ORIGINAL\\classifications.json'
BASE_PATH = 'C:\\Users\\mngra\\projects\\AI\\Pap\\PAP_DATA\\CRIC\\ORIGINAL'
IMAGE_INDEX = 150

# Modelos a comparar (comentar los que no tengas instalados)
MODELS_TO_TEST = [
    'biomedclip',
    # 'uni',       # Requiere descarga manual
    # 'optimus',   # Requiere HuggingFace access
    # 'uni2',      # Requiere descarga manual
]

# ============================================================================
# CARGAR IMAGEN
# ============================================================================

print("="*70)
print("COMPARACIÓN DE MODELOS FUNDACIONALES")
print("="*70)

print(f"\nCargando imagen {IMAGE_INDEX}...")
img, boxes, fname = load_image_and_boxes_from_json_cropped(
    JSON_PATH,
    BASE_PATH,
    index=IMAGE_INDEX,
    boxes_size=60,
    block_size=224
)
H, W = img.shape[:2]
print(f"✅ {fname} ({W}x{H})")
print(f"   Ground truth boxes: {len(boxes)}\n")

# ============================================================================
# PROCESAR CON CADA MODELO
# ============================================================================

results = {}

for model_name in MODELS_TO_TEST:
    print(f"\n{'='*70}")
    print(f"MODELO: {model_name.upper()}")
    print('='*70)
    
    try:
        # Cargar modelo
        print(f"Cargando {model_name}...")
        start_load = time.time()
        model = create_model(model_name, device='cpu')  # Cambiar a 'cuda' si tienes GPU
        load_time = time.time() - start_load
        print(f"✅ Cargado en {load_time:.2f}s")
        
        # Extraer embeddings
        print("Extrayendo embeddings...")
        start_embed = time.time()
        patch_data = get_all_patch_embeddings_from_image(img, model)
        embed_time = time.time() - start_embed
        print(f"✅ {len(patch_data)} patches en {embed_time:.2f}s")
        
        # Clustering básico (Fondo vs Tejido)
        print("Clustering (Fondo vs Tejido)...")
        start_cluster = time.time()
        clustered = run_block_clustering_on_embeddings(patch_data, n_clusters=2)
        cluster_time = time.time() - start_cluster
        print(f"✅ Clustering en {cluster_time:.2f}s")
        
        # Guardar resultados
        results[model_name] = {
            'load_time': load_time,
            'embed_time': embed_time,
            'cluster_time': cluster_time,
            'total_time': load_time + embed_time + cluster_time,
            'n_patches': len(patch_data),
            'embedding_dim': patch_data[0]['embedding'].shape[0] if patch_data else 0,
            'clustered_data': clustered
        }
        
        print(f"\n📊 RESUMEN {model_name}:")
        print(f"   Tiempo total: {results[model_name]['total_time']:.2f}s")
        print(f"   Dimensión embedding: {results[model_name]['embedding_dim']}")
        
    except Exception as e:
        print(f"⚠️ Error con {model_name}: {e}")
        results[model_name] = {'error': str(e)}

# ============================================================================
# COMPARAR RESULTADOS
# ============================================================================

print(f"\n{'='*70}")
print("COMPARACIÓN FINAL")
print('='*70)

print(f"\n{'Modelo':<15} {'Load (s)':<12} {'Embed (s)':<12} {'Total (s)':<12} {'Dim':<8}")
print("-"*70)

for model_name in MODELS_TO_TEST:
    if 'error' in results[model_name]:
        print(f"{model_name:<15} ERROR: {results[model_name]['error']}")
    else:
        r = results[model_name]
        print(f"{model_name:<15} {r['load_time']:<12.2f} {r['embed_time']:<12.2f} "
              f"{r['total_time']:<12.2f} {r['embedding_dim']:<8}")

# ============================================================================
# VISUALIZAR RESULTADOS
# ============================================================================

print(f"\n{'='*70}")
print("VISUALIZACIÓN")
print('='*70)

# Crear figura con subplots para cada modelo
successful_models = [m for m in MODELS_TO_TEST if 'error' not in results[m]]

if successful_models:
    n_models = len(successful_models)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, model_name in zip(axes, successful_models):
        clustered = results[model_name]['clustered_data']
        
        # Visualizar en el subplot
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        
        # Dibujar clusters
        from matplotlib.patches import Rectangle
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
        
        for patch in clustered:
            cluster_id = patch.get('cluster', 0)
            x1, y1, x2, y2 = patch['position']
            color = colors[cluster_id % len(colors)]
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=0.5, edgecolor=color, 
                           facecolor=color, alpha=0.3)
            ax.add_patch(rect)
        
        # Dibujar GT boxes
        for cls, x1, y1, x2, y2 in boxes:
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor='white', 
                           facecolor='none', linestyle='--')
            ax.add_patch(rect)
        
        ax.set_title(f"{model_name.upper()}\n"
                    f"Time: {results[model_name]['total_time']:.1f}s | "
                    f"Dim: {results[model_name]['embedding_dim']}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("\n✅ Visualización completada")
else:
    print("⚠️ No hay modelos exitosos para visualizar")

print(f"\n{'='*70}")
print("FIN DE LA COMPARACIÓN")
print('='*70)
