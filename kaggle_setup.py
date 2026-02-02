"""
Setup automático para Kaggle Notebooks.
Descarga dependencias, datasets y configura todo.

Ejecutar esto en la primera celda de tu Kaggle Notebook:
    %run kaggle_setup.py
"""

import os
import sys

print("⏳ Configurando Kaggle Notebook...")
print("="*60)

# ============================================================================
# 1. INSTALAR DEPENDENCIAS
# ============================================================================

print("\n[1/3] Instalando dependencias...")

import subprocess

packages = [
    'open-clip-torch==2.23.0',
    'transformers==4.35.2',
    'kagglehub',
]

for package in packages:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
        print(f"  ✅ {package}")
    except Exception as e:
        print(f"  ⚠️  Error instalando {package}: {e}")

# ============================================================================
# 2. DESCARGAR DATASET
# ============================================================================

print("\n[2/3] Descargando dataset (si es necesario)...")

try:
    import kagglehub
    
    # Intentar login (puede ser necesario en Kaggle)
    try:
        kagglehub.login()
        print("  ✅ Login a Kaggle")
    except:
        print("  ℹ️  Login a Kaggle (puede no ser necesario en Notebooks)")
    
    # Descargar dataset
    dataset_path = kagglehub.dataset_download('martingra/cric-dataset')
    print(f"  ✅ Dataset descargado: {dataset_path}")
    
except Exception as e:
    print(f"  ⚠️  Error descargando dataset: {e}")
    print("     Continúa si ya tienes el dataset en /kaggle/input/")

# ============================================================================
# 3. IMPORTAR MÓDULOS DEL PROYECTO
# ============================================================================

print("\n[3/3] Importando módulos...")

try:
    from config import KaggleConfig
    from main import run_pipeline
    from utils.image_processing import load_image_and_boxes_from_json_cropped
    from utils.embeddings import get_all_patch_embeddings_from_image
    from utils.multi_step_clustering import (
        run_block_clustering_on_embeddings,
        refinar_cluster_con_kmanes,
        decidir_fondo_vs_tejido,
        decidir_nucleos_vs_citoplasma
    )
    from utils.evaluation import evaluar_grupos_vs_boxes_plus
    from utils.visualization import visualizar_clusters_basicos
    
    print("  ✅ Módulos importados correctamente")
except Exception as e:
    print(f"  ⚠️  Error importando módulos: {e}")

# ============================================================================
# LISTO
# ============================================================================

print("\n" + "="*60)
print("✅ Setup completado")
print("="*60)

print("\n📖 PRÓXIMOS PASOS:")
print("""
1. Opción A - Ejecutar pipeline completo:
   
   cfg = KaggleConfig()
   metrics = run_pipeline(cfg)

2. Opción B - Ver tutorial paso a paso:
   
   exec(open('example.py').read())

3. Opción C - Procesar una sola imagen:
   
   from open_clip import create_model_from_pretrained, get_tokenizer
   model, preprocess = create_model_from_pretrained(
       'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
   )
   
   img, boxes, fname = load_image_and_boxes_from_json_cropped(
       '/kaggle/input/cric-dataset/classifications.json',
       '/kaggle/input/cric-dataset',
       index=150,
       boxes_size=60,
       block_size=224
   )
   
   patch_data = get_all_patch_embeddings_from_image(
       img, model, preprocess, tile_size=224
   )
""")

print("\n💡 TIP: Puedes usar Ctrl+Enter para ejecutar celdas rápidamente")
