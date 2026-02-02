"""
Pipeline de clustering multi-paso para segmentación de núcleos en imágenes citológicas.

Este script implementa 4 pasos de clustering jerárquico:
1. Fondo vs Tejido (KMeans k=2)
2. Núcleos vs Citoplasma (KMeans k=2 sobre tejido)
3. Núcleo en citoplasma vs Núcleo suelto (KMeans k=2 sobre núcleos)
4. Refinamiento final (KMeans k=2 sobre núcleos en citoplasma)

Luego limpia componentes pequeñas y agrupa parches en núcleos finales.
"""

import os
import json
import pickle
import traceback
from pathlib import Path

import numpy as np
import torch
from open_clip import create_model_from_pretrained, get_tokenizer

# Importar utilidades del proyecto
from utils.image_processing import (
    load_image_and_boxes_from_json_cropped,
    BETHESDA_CLASSES
)
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
# CONFIGURACIÓN PRINCIPAL
# ============================================================================

class Config:
    """Configuración del pipeline."""
    
    # Rutas de datos
    json_path = '/kaggle/input/cric-dataset/classifications.json'
    base_path = '/kaggle/input/cric-dataset'
    
    # Parámetros de la imagen
    box_size = 224                  # Tamaño de bloque para embeddings
    boxes_size_gt = 60              # Tamaño de box GT alrededor del centro
    
    # Parámetros de limpieza
    min_patches_componente = 3      # Mínimo de patches por componente conexa
    dilate_px_componentes = 32      # Dilatación para cerrar gaps
    
    # Imágenes a analizar
    images_to_analyze = [150, 151, 152, 153, 156, 157, 158, 159, 161, 163, 164, 
                         166, 168, 169, 170, 172, 179, 182, 200, 201, 210]
    
    # Evaluación
    match_mode = 'cover_gt'
    cover_gt_thr = 0.20
    
    # Output
    out_root = './eval_multistep'
    show_in_notebook = False
    save_figs = True


# ============================================================================
# HELPERS
# ============================================================================

def ensure_dir(path):
    """Crea directorio si no existe."""
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    """Guarda un objeto en JSON."""
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def save_pickle(obj, path):
    """Guarda un objeto en pickle."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def save_fig(path, dpi=120, close=True):
    """Guarda la figura actual."""
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    if close:
        plt.close(fig)


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_pipeline(cfg=None):
    """Ejecuta el pipeline completo."""
    if cfg is None:
        cfg = Config()
    
    ensure_dir(cfg.out_root)
    
    # Cargar modelo BiomedCLIP
    print("Cargando modelo BiomedCLIP...")
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    
    # Coleccionar métricas
    metrics_all = []
    
    # Procesar cada imagen
    for idx in cfg.images_to_analyze:
        img_dir = None
        try:
            # ===== Cargar imagen y GT =====
            print(f"\n{'='*60}")
            print(f"Procesando imagen {idx}...")
            print('='*60)
            
            img, boxes, fname = load_image_and_boxes_from_json_cropped(
                cfg.json_path,
                cfg.base_path,
                index=idx,
                boxes_size=cfg.boxes_size_gt,
                block_size=cfg.box_size
            )
            H, W = img.shape[:2]
            stem = os.path.splitext(fname)[0]
            img_dir = os.path.join(cfg.out_root, f"{idx:04d}_{stem}")
            ensure_dir(img_dir)
            
            print(f"  Imagen: {fname} ({W}x{H})")
            print(f"  GT boxes: {len(boxes)}")
            
            # ===== PASO 0: Extraer embeddings =====
            print(f"  [0] Extrayendo embeddings...")
            patch_data = get_all_patch_embeddings_from_image(
                img, model, preprocess, tile_size=cfg.box_size, stride=cfg.box_size
            )
            print(f"      Patches extraídos: {len(patch_data)}")
            
            # ===== PASO 1: Fondo vs Tejido =====
            print(f"  [1] Clustering: Fondo vs Tejido...")
            clustered_k2 = run_block_clustering_on_embeddings(patch_data, method='kmeans', n_clusters=2)
            fondo_id, tejido_id, _ = decidir_fondo_vs_tejido(img, clustered_k2, cluster_field='cluster')
            
            if cfg.save_figs:
                visualizar_clusters_basicos(img, clustered_k2, boxes=boxes, cluster_field='cluster',
                                           grid_size=cfg.box_size, title="Paso 1: Fondo vs Tejido")
                save_fig(os.path.join(img_dir, "01_fondo_vs_tejido.png"))
            
            # ===== PASO 2: Núcleos vs Citoplasma =====
            print(f"  [2] Clustering: Núcleos vs Citoplasma (sobre tejido)...")
            paso_2 = refinar_cluster_con_kmeans(
                clustered_k2, cluster_id=tejido_id, nuevo_k=2,
                cluster_field='cluster', new_field='subcluster'
            )
            nucleos_id, cyto_id, _ = decidir_nucleos_vs_citoplasma(img, paso_2, cluster_field='subcluster')
            
            if cfg.save_figs:
                visualizar_clusters_basicos(img, paso_2, boxes=boxes, cluster_field='subcluster',
                                           title="Paso 2: Núcleos vs Citoplasma")
                save_fig(os.path.join(img_dir, "02_nucleos_vs_cyto.png"))
            
            # ===== PASO 3: Núcleo en citoplasma vs Suelto =====
            print(f"  [3] Clustering: Núcleo en citoplasma vs Suelto (sobre núcleos)...")
            paso_3 = refinar_cluster_con_kmeans(
                paso_2, cluster_id=nucleos_id, nuevo_k=2,
                cluster_field='subcluster', new_field='subsubcluster'
            )
            nucleo_en_cito_id, nucleo_suelto_id, _ = decidir_nucleos_vs_citoplasma(
                img, paso_3, cluster_field='subsubcluster'
            )
            
            if cfg.save_figs:
                visualizar_clusters_basicos(img, paso_3, boxes=boxes, cluster_field='subsubcluster',
                                           title="Paso 3: Núcleo en Citoplasma vs Suelto")
                save_fig(os.path.join(img_dir, "03_nucleo_cito_vs_suelto.png"))
            
            # ===== PASO 4: Refinamiento final =====
            print(f"  [4] Clustering: Refinamiento de núcleos en citoplasma...")
            paso_4 = refinar_cluster_con_kmeans(
                paso_3, cluster_id=nucleo_en_cito_id, nuevo_k=2,
                cluster_field='subsubcluster', new_field='subsubsubcluster'
            )
            nucleo_objetivo_id, _, _ = decidir_nucleos_vs_citoplasma(
                img, paso_4, cluster_field='subsubsubcluster'
            )
            
            if cfg.save_figs:
                visualizar_clusters_basicos(img, paso_4, boxes=boxes, cluster_field='subsubsubcluster',
                                           title="Paso 4: Refinamiento Final")
                save_fig(os.path.join(img_dir, "04_refinamiento_final.png"))
            
            # ===== Seleccionar patches objetivo =====
            salida_multistep = [p for p in paso_4 if p.get('subsubsubcluster') == nucleo_objetivo_id]
            print(f"  Patches objetivo (pre-limpieza): {len(salida_multistep)}")
            
            # ===== Limpiar componentes pequeñas =====
            print(f"  [5] Limpiando componentes pequeñas...")
            kept, removed, dbg_comp = limpiar_patches_por_componentes_mask(
                img, salida_multistep,
                min_patches=cfg.min_patches_componente,
                connectivity=8,
                dilate_px=cfg.dilate_px_componentes
            )
            print(f"      Antes: {len(salida_multistep)} | Después: {len(kept)} | "
                  f"Removidos: {len(removed)}")
            
            if cfg.save_figs:
                visualizar_limpieza_patches(img, kept, removed, boxes=boxes,
                                           title="Limpieza de componentes")
                save_fig(os.path.join(img_dir, "05_limpieza.png"))
            
            # ===== Agrupar patches en núcleos =====
            print(f"  [6] Agrupando patches...")
            grupos, dbg_groups = agrupar_patches_en_grupos(
                img, kept,
                min_patches_por_grupo=cfg.min_patches_componente,
                dilate_px=0,
                connectivity=8
            )
            print(f"      Grupos detectados: {len(grupos)}")
            
            # ===== Evaluación =====
            print(f"  [7] Evaluando resultados...")
            metrics = evaluar_grupos_vs_boxes_plus(
                img, grupos, boxes,
                match_mode=cfg.match_mode,
                cover_gt_thr=cfg.cover_gt_thr
            )
            
            if cfg.save_figs:
                visualizar_grupos_vs_boxes(img, grupos, boxes,
                                          match_mode=cfg.match_mode,
                                          cover_gt_thr=cfg.cover_gt_thr)
                save_fig(os.path.join(img_dir, "06_grupos_vs_GT.png"))
            
            # ===== Guardar resultados =====
            metrics_row = {
                'index': int(idx),
                'fname': str(fname),
                'H': int(H), 'W': int(W),
                'groups_TP': int(metrics.get('groups_TP', 0)),
                'groups_FP': int(metrics.get('groups_FP', 0)),
                'gt_hit': int(metrics.get('gt_hit', 0)),
                'gt_total': int(metrics.get('gt_total', 0)),
                'group_precision': float(metrics.get('group_precision', 0.0)),
                'gt_recall_coverage': float(metrics.get('gt_recall_coverage', 0.0)),
                'f1_coverage': float(metrics.get('f1_coverage', 0.0)),
                'match_mode': cfg.match_mode,
                'cover_gt_thr': float(cfg.cover_gt_thr),
                'n_patches_final': int(len(kept)),
                'n_grupos': int(len(grupos))
            }
            
            save_json(metrics_row, os.path.join(img_dir, "metrics.json"))
            metrics_all.append(metrics_row)
            
            print(f"  ✅ OK")
            print(f"     TP={metrics_row['groups_TP']} FP={metrics_row['groups_FP']} "
                  f"GT={metrics_row['gt_hit']}/{metrics_row['gt_total']} "
                  f"F1={metrics_row['f1_coverage']:.3f}")
        
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            if img_dir:
                with open(os.path.join(img_dir, "ERROR.txt"), "w") as f:
                    f.write(f"{e}\n\n{traceback.format_exc()}")
    
    # ===== Resumen global =====
    summary_path = os.path.join(cfg.out_root, "metrics_summary.json")
    save_json(metrics_all, summary_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Pipeline completo")
    print(f"{'='*60}")
    print(f"Resumen guardado en: {summary_path}")
    print(f"Total imágenes procesadas: {len(metrics_all)}")
    
    if metrics_all:
        avg_f1 = np.mean([m['f1_coverage'] for m in metrics_all])
        print(f"F1 promedio: {avg_f1:.3f}")
    
    return metrics_all


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    cfg = Config()
    metrics = run_pipeline(cfg)
