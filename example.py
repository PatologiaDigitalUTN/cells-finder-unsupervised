"""
Script de ejemplo: Clustering de una imagen paso a paso.

Este archivo muestra cómo usar el pipeline de forma interactiva,
procesando una imagen completa con todos los pasos.
"""

import matplotlib.pyplot as plt
from open_clip import create_model_from_pretrained, get_tokenizer

from utils.image_processing import load_image_and_boxes_from_json_cropped, BETHESDA_CLASSES
from utils.embeddings import get_all_patch_embeddings_from_image
from utils.multi_step_clustering import (
    run_block_clustering_on_embeddings,
    refinar_cluster_con_kmeans,
    decidir_fondo_vs_tejido,
    decidir_nucleos_vs_citoplasma
)
from utils.evaluation import (
    limpiar_patches_por_componentes_mask,
    agrupar_patches_en_grupos,
    evaluar_grupos_vs_boxes_plus
)
from utils.visualization import (
    visualizar_clusters_basicos,
    visualizar_limpieza_patches,
    visualizar_grupos_vs_boxes
)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Cambiar estas rutas según tus datos
JSON_PATH = '/kaggle/input/cric-dataset/classifications.json'
BASE_PATH = '/kaggle/input/cric-dataset'

# Índice de la imagen a procesar
IMAGE_INDEX = 150

# Parámetros
BOX_SIZE = 224          # Tamaño de tile para embeddings
BOXES_SIZE_GT = 60      # Tamaño de box GT

# ============================================================================
# PASO 0: CARGA DE MODELO
# ============================================================================

print("Cargando modelo BiomedCLIP...")
model, preprocess = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer = get_tokenizer(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
print("✅ Modelo cargado")

# ============================================================================
# PASO 1: CARGAR IMAGEN Y GROUND TRUTH
# ============================================================================

print(f"\nCargando imagen {IMAGE_INDEX}...")
img, boxes, fname = load_image_and_boxes_from_json_cropped(
    JSON_PATH,
    BASE_PATH,
    index=IMAGE_INDEX,
    boxes_size=BOXES_SIZE_GT,
    block_size=BOX_SIZE
)
H, W = img.shape[:2]
print(f"  Imagen: {fname}")
print(f"  Tamaño: {W}x{H}")
print(f"  Ground truth boxes: {len(boxes)}")

# Mostrar imagen con boxes GT
print("\n→ Mostrando imagen con boxes GT...")
from utils.visualization import mostrar_imagen_con_boxes
mostrar_imagen_con_boxes(img, boxes, title=f"Imagen {IMAGE_INDEX}: {fname}")

# ============================================================================
# PASO 2: EXTRAER EMBEDDINGS
# ============================================================================

print("\nExtrayendo embeddings de patches...")
patch_data = get_all_patch_embeddings_from_image(
    img, model, preprocess, tile_size=BOX_SIZE, stride=BOX_SIZE
)
print(f"  Patches extraídos: {len(patch_data)}")

# ============================================================================
# PASO 3: CLUSTERING PASO 1 - FONDO vs TEJIDO
# ============================================================================

print("\n" + "="*60)
print("PASO 1: Clustering Fondo vs Tejido")
print("="*60)

clustered_k2 = run_block_clustering_on_embeddings(
    patch_data, method='kmeans', n_clusters=2
)
print(f"Clusters asignados")

fondo_id, tejido_id, stats1 = decidir_fondo_vs_tejido(
    img, clustered_k2, cluster_field='cluster'
)

print(f"\n→ Mostrando clustering paso 1...")
visualizar_clusters_basicos(
    img, clustered_k2, boxes=boxes, cluster_field='cluster',
    grid_size=BOX_SIZE, title="Paso 1: Fondo vs Tejido"
)

# ============================================================================
# PASO 4: CLUSTERING PASO 2 - NÚCLEOS vs CITOPLASMA
# ============================================================================

print("\n" + "="*60)
print("PASO 2: Clustering Núcleos vs Citoplasma (sobre tejido)")
print("="*60)

paso_2 = refinar_cluster_con_kmeans(
    clustered_k2,
    cluster_id=tejido_id,
    nuevo_k=2,
    cluster_field='cluster',
    new_field='subcluster'
)
print(f"Sub-clustering sobre tejido completado")

nucleos_id, cyto_id, stats2 = decidir_nucleos_vs_citoplasma(
    img, paso_2, cluster_field='subcluster'
)

print(f"\n→ Mostrando clustering paso 2...")
visualizar_clusters_basicos(
    img, paso_2, boxes=boxes, cluster_field='subcluster',
    title="Paso 2: Núcleos vs Citoplasma"
)

# ============================================================================
# PASO 5: CLUSTERING PASO 3 - NÚCLEO EN CITOPLASMA vs SUELTO
# ============================================================================

print("\n" + "="*60)
print("PASO 3: Clustering Núcleo en Citoplasma vs Suelto (sobre núcleos)")
print("="*60)

paso_3 = refinar_cluster_con_kmeans(
    paso_2,
    cluster_id=nucleos_id,
    nuevo_k=2,
    cluster_field='subcluster',
    new_field='subsubcluster'
)
print(f"Sub-clustering sobre núcleos completado")

nucleo_en_cito_id, nucleo_suelto_id, stats3 = decidir_nucleos_vs_citoplasma(
    img, paso_3, cluster_field='subsubcluster'
)

print(f"\n→ Mostrando clustering paso 3...")
visualizar_clusters_basicos(
    img, paso_3, boxes=boxes, cluster_field='subsubcluster',
    title="Paso 3: Núcleo en Citoplasma vs Suelto"
)

# ============================================================================
# PASO 6: CLUSTERING PASO 4 - REFINAMIENTO FINAL
# ============================================================================

print("\n" + "="*60)
print("PASO 4: Refinamiento de Núcleos en Citoplasma")
print("="*60)

paso_4 = refinar_cluster_con_kmeans(
    paso_3,
    cluster_id=nucleo_en_cito_id,
    nuevo_k=2,
    cluster_field='subsubcluster',
    new_field='subsubsubcluster'
)
print(f"Refinamiento final completado")

nucleo_objetivo_id, _, stats4 = decidir_nucleos_vs_citoplasma(
    img, paso_4, cluster_field='subsubsubcluster'
)

print(f"\n→ Mostrando clustering paso 4...")
visualizar_clusters_basicos(
    img, paso_4, boxes=boxes, cluster_field='subsubsubcluster',
    title="Paso 4: Refinamiento Final"
)

# ============================================================================
# PASO 7: SELECCIONAR PATCHES OBJETIVO
# ============================================================================

print("\n" + "="*60)
print("Seleccionando patches objetivo")
print("="*60)

salida_multistep = [p for p in paso_4 if p.get('subsubsubcluster') == nucleo_objetivo_id]
print(f"Patches objetivo (pre-limpieza): {len(salida_multistep)}")

# ============================================================================
# PASO 8: LIMPIAR COMPONENTES PEQUEÑAS
# ============================================================================

print("\n" + "="*60)
print("Limpiando componentes pequeñas")
print("="*60)

kept, removed, dbg_comp = limpiar_patches_por_componentes_mask(
    img,
    salida_multistep,
    min_patches=3,
    connectivity=8,
    dilate_px=32
)

print(f"Antes limpieza: {len(salida_multistep)}")
print(f"Después limpieza: {len(kept)}")
print(f"Removidos: {len(removed)}")
print(f"Distribución de componentes: {dbg_comp['label_counts_patches']}")

print(f"\n→ Mostrando resultado de limpieza...")
visualizar_limpieza_patches(
    img, kept, removed, boxes=boxes,
    title="Resultado tras limpieza"
)

# ============================================================================
# PASO 9: AGRUPAR PATCHES EN NÚCLEOS
# ============================================================================

print("\n" + "="*60)
print("Agrupando patches en núcleos finales")
print("="*60)

grupos, dbg_groups = agrupar_patches_en_grupos(
    img,
    kept,
    min_patches_por_grupo=3,
    dilate_px=0,
    connectivity=8
)

print(f"Grupos detectados: {len(grupos)}")
print(f"Componentes totales: {dbg_groups['num_components']}")
print(f"Distribución: {dbg_groups['label_patch_counts']}")

# ============================================================================
# PASO 10: EVALUACIÓN
# ============================================================================

print("\n" + "="*60)
print("Evaluación Final")
print("="*60)

metrics = evaluar_grupos_vs_boxes_plus(
    img, grupos, boxes,
    match_mode='cover_gt',
    cover_gt_thr=0.20
)

print(f"\nMétricas:")
print(f"  TP (grupos): {metrics['groups_TP']}")
print(f"  FP (grupos): {metrics['groups_FP']}")
print(f"  GT cubiertos: {metrics['gt_hit']}/{metrics['gt_total']}")
print(f"  Precisión: {metrics['group_precision']:.3f}")
print(f"  Recall (cobertura): {metrics['gt_recall_coverage']:.3f}")
print(f"  F1: {metrics['f1_coverage']:.3f}")

print(f"\n→ Mostrando resultado final vs GT...")
visualizar_grupos_vs_boxes(
    img, grupos, boxes,
    match_mode='cover_gt',
    cover_gt_thr=0.20
)

print("\n" + "="*60)
print("✅ Pipeline completado exitosamente")
print("="*60)
