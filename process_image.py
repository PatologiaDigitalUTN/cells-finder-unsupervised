"""
Pipeline exploración de clustering - Árbol de decisión adaptativo.

Este script ejecuta un pipeline exploratorio donde cada paso divide clusters en k=2
sin usar funciones de decisión. Crea un árbol de clustering y evalúa todas las ramas.

Ejecuta:
  python process_image.py --json <json> --images_dir <dir> --index <idx> --n_steps <n> [opciones]

Ejemplo:
  python process_image.py \\
    --json "C:\\data\\_annotations.coco.json" \\
    --images_dir "C:\\data" \\
    --index 0 \\
    --n_steps 4 \\
    --preprocessing clahe \\
    --visualize

Funcionalidad:
  - Paso 0: Kmeans(k=2) en todos los embeddings (Fondo vs Tejido)
  - Pasos 1+: Kmeans(k=2) en CADA rama anterior
  - Resultado: Árbol binario con N ramas
  - Métricas: Precisión, Recall, F1 para cada rama (no para paso 0)
  - Reporte: Mostra la mejor rama según F1 score
"""

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Importar módulos del proyecto
from utils.coco_loader import list_coco_images, get_coco_categories, load_image_and_boxes_from_coco
from utils.image_processing import apply_preprocessing
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
    evaluar_grupos_vs_boxes_plus,
    _to_xyxy,
    _overlap_area
)
from utils.visualization import visualizar_clusters_basicos, visualizar_limpieza_patches
from utils.model_factory import create_model, BaseEmbeddingModel


class ClusterNode:
    """Nodo en el árbol de clustering binario."""
    
    def __init__(self, step, node_id, patches, cluster_field='cluster'):
        """
        Parameters:
        -----------
        step : int
            Número de paso (0 para root)
        node_id : str
            ID único del nodo (ej: "0_0", "1_0L", "1_0R")
        patches : list
            Lista de patches con embeddings
        cluster_field : str
            Campo donde se almacena el ID del cluster
        """
        self.step = step
        self.node_id = node_id
        self.patches = patches
        self.cluster_field = cluster_field
        self.n_patches = len(patches)
        
        # Información de clustering
        self.cluster_id_0 = None
        self.cluster_id_1 = None
        self.child_left = None  # Nodo hijo (cluster 0)
        self.child_right = None  # Nodo hijo (cluster 1)
        
        # Métricas (solo para pasos > 0)
        self.metrics = None
    
    def is_leaf(self):
        """Retorna True si es una hoja (sin hijos)."""
        return self.child_left is None and self.child_right is None
    
    def get_all_leaves(self):
        """Retorna todas las hojas del subárbol."""
        if self.is_leaf():
            return [self]
        leaves = []
        if self.child_left:
            leaves.extend(self.child_left.get_all_leaves())
        if self.child_right:
            leaves.extend(self.child_right.get_all_leaves())
        return leaves


class ClusteringTree:
    """Árbol de clustering que crece en pasos, dividiendo cada rama en k=2."""
    
    def __init__(self, patch_data, img, ground_truth, H, W, evaluation_strategy):
        """
        Inicializa el árbol con todos los embeddings como raíz.
        
        Parameters:
        -----------
        patch_data : list
            Lista de patches con embeddings
        img : np.ndarray
            Imagen para visualización
        ground_truth : Any
            Ground truth en formato de la estrategia
        H, W : int
            Dimensiones de la imagen
        evaluation_strategy : EvaluationStrategy
            Estrategia de evaluación a usar
        """
        self.patch_data = patch_data
        self.img = img
        self.ground_truth = ground_truth
        self.H = H
        self.W = W
        self.evaluation_strategy = evaluation_strategy
        
        # Inicializar raíz
        self.root = ClusterNode(step=0, node_id="root", patches=patch_data)
        self.all_nodes = [self.root]
        self.step_history = []
    
    def grow_step(self, visualize=True):
        """
        Crece el árbol un paso: divide todas las hojas actuales en k=2.
        Calcula métricas para cada rama (excepto paso 0).
        """
        current_step = self.root.get_all_leaves()[0].step + 1
        print(f"\n{'='*70}")
        print(f"PASO {current_step}: División de ramas")
        print('='*70)
        
        leaves = self.root.get_all_leaves()
        print(f"Dividiendo {len(leaves)} ramas...\n")
        
        new_nodes = []
        step_results = []
        
        for leaf in leaves:
            if leaf.n_patches < 2:
                print(f"  ⚠️ Nodo {leaf.node_id}: muy pocos patches ({leaf.n_patches}), omitido")
                continue
            
            # Aplicar kmeans k=2 en los patches de esta hoja
            clustered = run_block_clustering_on_embeddings(leaf.patches, n_clusters=2)
            
            # Extraer los dos clusters basándose en el campo 'cluster' que agrega la función
            cluster_ids = set(p.get('cluster') for p in clustered if 'cluster' in p)
            cluster_ids = sorted(list(cluster_ids))
            
            if len(cluster_ids) != 2:
                print(f"  ⚠️ Nodo {leaf.node_id}: no se dividió en 2 clusters")
                continue
            
            cluster_0_id, cluster_1_id = cluster_ids[0], cluster_ids[1]
            patches_0 = [p for p in clustered if p.get('cluster') == cluster_0_id]
            patches_1 = [p for p in clustered if p.get('cluster') == cluster_1_id]
            
            # Crear nodos hijos
            node_left = ClusterNode(
                step=current_step,
                node_id=f"{leaf.node_id}_L",
                patches=patches_0,
                cluster_field='cluster'  # Usar siempre 'cluster' como campo
            )
            node_right = ClusterNode(
                step=current_step,
                node_id=f"{leaf.node_id}_R",
                patches=patches_1,
                cluster_field='cluster'
            )
            
            node_left.cluster_id_0 = cluster_0_id
            node_right.cluster_id_1 = cluster_1_id
            
            leaf.child_left = node_left
            leaf.child_right = node_right
            
            new_nodes.extend([node_left, node_right])
            
            # Calcular métricas si paso > 0
            if current_step > 0:
                metrics_left = self._evaluate_node(node_left)
                metrics_right = self._evaluate_node(node_right)
                
                node_left.metrics = metrics_left
                node_right.metrics = metrics_right
                
                step_results.append({
                    'node': node_left.node_id,
                    'n_patches': node_left.n_patches,
                    'f1': metrics_left['f1_coverage']
                })
                step_results.append({
                    'node': node_right.node_id,
                    'n_patches': node_right.n_patches,
                    'f1': metrics_right['f1_coverage']
                })
                
                print(f"  ✅ {node_left.node_id}: {node_left.n_patches} patches | F1: {metrics_left['f1_coverage']:.3f}")
                print(f"  ✅ {node_right.node_id}: {node_right.n_patches} patches | F1: {metrics_right['f1_coverage']:.3f}")
            else:
                print(f"  ✅ {node_left.node_id}: {node_left.n_patches} patches")
                print(f"  ✅ {node_right.node_id}: {node_right.n_patches} patches")
        
        self.all_nodes.extend(new_nodes)
        self.step_history.append(step_results)
        
        print(f"\n  Total de nodos ahora: {len(self.all_nodes)}")
    
    def _evaluate_node(self, node):
        """Evalúa un nodo usando la estrategia de evaluación configurada."""
        # Evaluar patches con estrategia
        metrics = self.evaluation_strategy.evaluate_patches(
            node.patches,
            self.img,
            self.ground_truth
        )
        
        return metrics
    
    def get_best_node(self):
        """Retorna el nodo con el mejor F1 score de TODOS los pasos (no solo hojas)."""
        nodes_with_metrics = [n for n in self.all_nodes if n.metrics is not None]
        
        if not nodes_with_metrics:
            return None
        
        best_node = max(nodes_with_metrics, key=lambda x: x.metrics['f1_coverage'])
        return best_node
    
    def get_best_leaf(self):
        """Retorna la hoja (nodo final) con el mejor F1 score."""
        leaves = self.root.get_all_leaves()
        
        # Filtrar hojas con métricas
        leaves_with_metrics = [l for l in leaves if l.metrics is not None]
        
        if not leaves_with_metrics:
            return None
        
        best_leaf = max(leaves_with_metrics, key=lambda x: x.metrics['f1_coverage'])
        return best_leaf
    
    def get_optimal_steps(self):
        """Retorna el número de pasos óptimo basado en dónde se encuentra el mejor nodo."""
        best = self.get_best_node()
        
        if best is None:
            return None
        
        return best.step
    
    def visualize_best_node_with_grouping(self):
        """Visualiza el mejor nodo usando la estrategia de evaluación configurada."""
        best = self.get_best_node()
        
        if best is None or best.metrics is None:
            print("⚠️ No hay rama con métricas para visualizar")
            return
        
        print("\n📸 VISUALIZANDO MEJOR RAMA:\n")
        
        # Obtener datos de visualización de la estrategia
        viz_data = self.evaluation_strategy.get_visualization_data(
            best.patches, 
            self.img, 
            self.ground_truth
        )
        
        print(f"Patches en mejor rama: {best.n_patches}")
        
        # Visualización
        fig, ax = plt.subplots(figsize=(14, 12))
        
        if self.img.ndim == 2:
            ax.imshow(self.img, cmap='gray')
        else:
            ax.imshow(self.img)
        
        # Renderizar según tipo de visualización
        if viz_data['type'] == 'bbox':
            # Estrategia de bounding boxes
            grupos = viz_data['groups']
            kept = viz_data['kept_patches']
            
            print(f"Patches después de limpieza: {len(kept)}")
            print(f"Núcleos detectados: {len(grupos)}\n")
            
            # Dibujar grupos (TP en verde, FP en rojo)
            for i, grupo in enumerate(grupos):
                x1, y1, x2, y2 = grupo['position']
                is_tp = best.metrics['pred_hits'][i] if i < len(best.metrics['pred_hits']) else False
                color = (0.2, 0.8, 0.2) if is_tp else (0.8, 0.2, 0.2)
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor=color, alpha=0.3
                )
                ax.add_patch(rect)
            
            # Dibujar GT boxes
            boxes = viz_data['gt_boxes']
            gt_covered = viz_data['gt_covered']
            for gt_idx, (cls, x1, y1, x2, y2) in enumerate(boxes):
                color = 'blue' if gt_covered[gt_idx] else 'orange'
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2.5, edgecolor=color, facecolor='none', linestyle='--'
                )
                ax.add_patch(rect)
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                ax.plot(cx, cy, 'o', markersize=10, color=color,
                       markeredgecolor='yellow', markeredgewidth=2)
            
            legend_elements = [
                Rectangle((0, 0), 1, 1, fc=(0.2, 0.8, 0.2), alpha=0.5, label='TP (Correcto)'),
                Rectangle((0, 0), 1, 1, fc=(0.8, 0.2, 0.2), alpha=0.5, label='FP (Incorrecto)'),
                Rectangle((0, 0), 1, 1, fc='none', edgecolor='blue', linewidth=2,
                         linestyle='--', label='GT cubierto'),
                Rectangle((0, 0), 1, 1, fc='none', edgecolor='orange', linewidth=2,
                         linestyle='--', label='GT no cubierto'),
            ]
            
            title = (f"🏆 MEJOR NODO OVERALL: {best.node_id} (Paso {best.step})\n"
                    f"TP={best.metrics['groups_TP']} | FP={best.metrics['groups_FP']} | "
                    f"F1={best.metrics['f1_coverage']:.3f}")
        
        elif viz_data['type'] == 'segmentation':
            # Estrategia de segmentación
            print(f"Métricas pixel-wise calculadas\n")
            
            # Overlay de máscaras TP (verde), FP (rojo), FN (azul)
            overlay = np.zeros((*self.img.shape[:2], 3), dtype=np.uint8)
            
            tp_mask = viz_data['tp_mask']
            fp_mask = viz_data['fp_mask']
            fn_mask = viz_data['fn_mask']
            
            overlay[tp_mask > 0] = [0, 255, 0]     # Verde: TP
            overlay[fp_mask > 0] = [255, 0, 0]     # Rojo: FP  
            overlay[fn_mask > 0] = [0, 0, 255]     # Azul: FN
            
            ax.imshow(overlay, alpha=0.4)
            
            # Contornos de GT
            gt_masks = viz_data['gt_masks']
            for gt_mask in gt_masks:
                contours = cv2.findContours(
                    gt_mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )[0]
                for contour in contours:
                    contour = contour.squeeze()
                    if contour.ndim == 2:
                        ax.plot(contour[:, 0], contour[:, 1], 
                               color='yellow', linewidth=2, linestyle='--')
            
            legend_elements = [
                Rectangle((0, 0), 1, 1, fc=(0, 1, 0), alpha=0.5, label='TP (Verdaderos Positivos)'),
                Rectangle((0, 0), 1, 1, fc=(1, 0, 0), alpha=0.5, label='FP (Falsos Positivos)'),
                Rectangle((0, 0), 1, 1, fc=(0, 0, 1), alpha=0.5, label='FN (Falsos Negativos)'),
                Rectangle((0, 0), 1, 1, fc='none', edgecolor='yellow', linewidth=2,
                         linestyle='--', label='GT mask'),
            ]
            
            dice = best.metrics.get('dice', 0.0)
            iou = best.metrics.get('iou', 0.0)
            title = (f"🏆 MEJOR NODO OVERALL: {best.node_id} (Paso {best.step})\n"
                    f"DICE={dice:.3f} | IoU={iou:.3f} | F1={best.metrics['f1_coverage']:.3f}")
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    
    def print_summary(self):
        """Imprime un resumen del árbol y sus métricas."""
        print("\n" + "="*70)
        print("RESUMEN DEL ÁRBOL DE CLUSTERING")
        print("="*70)
        
        leaves = self.root.get_all_leaves()
        print(f"\nTotal de ramas finales: {len(leaves)}")
        
        # Mostrar todas las hojas con sus métricas
        print("\n📊 MÉTRICAS POR RAMA:")
        leaves_with_metrics = [(l.node_id, l.metrics) for l in leaves if l.metrics is not None]
        leaves_with_metrics.sort(key=lambda x: x[1]['f1_coverage'], reverse=True)
        
        for node_id, metrics in leaves_with_metrics:
            print(f"\n  {node_id}:")
            
            # Detectar tipo de estrategia basándose en las métricas disponibles
            if 'groups_TP' in metrics:
                # Estrategia bbox
                print(f"    TP: {metrics['groups_TP']:2d} | FP: {metrics['groups_FP']:2d}")
                print(f"    Precisión: {metrics['group_precision']:.3f} | Recall: {metrics['gt_recall_coverage']:.3f}")
            elif 'dice' in metrics:
                # Estrategia segmentation
                print(f"    DICE: {metrics['dice']:.3f} | IoU: {metrics['iou']:.3f}")
                print(f"    Precisión: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}")
            
            print(f"    F1 Score: {metrics['f1_coverage']:.3f}")
        
        # Mostrar la mejor rama
        best = self.get_best_leaf()
        if best and best.metrics:
            print(f"\n🏆 MEJOR RAMA: {best.node_id}")
            print(f"   F1 Score: {best.metrics['f1_coverage']:.3f}")
            print(f"   Patches: {best.n_patches}")
    
    def get_metrics_by_step(self):
        """Retorna estadísticas de métricas agrupadas por paso."""
        metrics_by_step = {}
        
        # Agrupar nodos por paso
        for node in self.all_nodes:
            if node.step not in metrics_by_step:
                metrics_by_step[node.step] = {'nodes': [], 'f1_scores': []}
            
            metrics_by_step[node.step]['nodes'].append(node)
            
            if node.metrics is not None:
                metrics_by_step[node.step]['f1_scores'].append(node.metrics['f1_coverage'])
        
        # Calcular estadísticas por paso
        step_stats = {}
        for step in sorted(metrics_by_step.keys()):
            f1_scores = metrics_by_step[step]['f1_scores']
            n_nodes = len(metrics_by_step[step]['nodes'])
            
            if f1_scores:
                step_stats[step] = {
                    'n_nodes': n_nodes,
                    'n_evaluated': len(f1_scores),
                    'f1_min': min(f1_scores),
                    'f1_max': max(f1_scores),
                    'f1_mean': sum(f1_scores) / len(f1_scores),
                    'f1_std': (sum((x - (sum(f1_scores) / len(f1_scores)))**2 for x in f1_scores) / len(f1_scores))**0.5 if len(f1_scores) > 1 else 0
                }
            else:
                step_stats[step] = {
                    'n_nodes': n_nodes,
                    'n_evaluated': 0,
                    'f1_min': None,
                    'f1_max': None,
                    'f1_mean': None,
                    'f1_std': None
                }
        
        return step_stats
    
    def visualize_tree_structure(self):
        """Visualiza el árbol completo con nodos y sus F1 scores."""
        step_stats = self.get_metrics_by_step()
        
        # Crear figura para visualizar estructura
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        
        # ===== GRÁFICO 1: Estadísticas por paso =====
        ax = axes[0]
        
        steps = sorted([s for s in step_stats.keys()])
        f1_maxes = []
        f1_means = []
        f1_mins = []
        n_nodes_list = []
        
        for step in steps:
            stats = step_stats[step]
            f1_maxes.append(stats['f1_max'] if stats['f1_max'] is not None else 0)
            f1_means.append(stats['f1_mean'] if stats['f1_mean'] is not None else 0)
            f1_mins.append(stats['f1_min'] if stats['f1_min'] is not None else 0)
            n_nodes_list.append(stats['n_nodes'])
        
        x = np.arange(len(steps))
        width = 0.25
        
        ax.bar(x - width, f1_mins, width, label='F1 Mínimo', alpha=0.8, color='#ff7f0e')
        ax.bar(x, f1_means, width, label='F1 Promedio', alpha=0.8, color='#2ca02c')
        ax.bar(x + width, f1_maxes, width, label='F1 Máximo', alpha=0.8, color='#1f77b4')
        
        ax.set_xlabel('Paso del Árbol', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Evolución de Métricas por Paso', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Paso {s}' for s in steps])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Agregar etiquetas con n_nodes
        for i, (step, n_nodes) in enumerate(zip(steps, n_nodes_list)):
            if step > 0:  # No mostrar en paso 0 que no tiene métricas
                ax.text(i, 1.02, f'n={n_nodes}', ha='center', fontsize=9, color='gray')
        
        # ===== GRÁFICO 2: Tabla de estadísticas =====
        ax = axes[1]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        table_data.append(['Paso', 'Nodos', 'Evaluados', 'F1 Min', 'F1 Media', 'F1 Max'])
        
        for step in steps:
            stats = step_stats[step]
            
            f1_min_str = f"{stats['f1_min']:.3f}" if stats['f1_min'] is not None else "—"
            f1_mean_str = f"{stats['f1_mean']:.3f}" if stats['f1_mean'] is not None else "—"
            f1_max_str = f"{stats['f1_max']:.3f}" if stats['f1_max'] is not None else "—"
            
            table_data.append([
                f"Paso {step}",
                str(stats['n_nodes']),
                str(stats['n_evaluated']),
                f1_min_str,
                f1_mean_str,
                f1_max_str
            ])
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.12, 0.12, 0.15, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Colorear header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Colorear filas alternadas
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
                else:
                    table[(i, j)].set_facecolor('#F2F2F2')
        
        ax.set_title('Estadísticas de Métricas por Paso', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir tabla en consola también
        print("\n" + "="*80)
        print("ANÁLISIS DE MÉTRICAS POR PASO")
        print("="*80)
        print(f"{'Paso':<6} {'Nodos':<8} {'Evaluados':<12} {'F1 Min':<10} {'F1 Media':<10} {'F1 Max':<10}")
        print("-"*80)
        
        for step in steps:
            stats = step_stats[step]
            f1_min_str = f"{stats['f1_min']:.3f}" if stats['f1_min'] is not None else "—"
            f1_mean_str = f"{stats['f1_mean']:.3f}" if stats['f1_mean'] is not None else "—"
            f1_max_str = f"{stats['f1_max']:.3f}" if stats['f1_max'] is not None else "—"
            
            print(f"{step:<6} {stats['n_nodes']:<8} {stats['n_evaluated']:<12} "
                  f"{f1_min_str:<10} {f1_mean_str:<10} {f1_max_str:<10}")
        
        print("="*80)


class ImageProcessingPipeline:
    """Pipeline que crea un árbol de clustering explorador."""
    
    def __init__(self, json_path, images_dir, model_name='biomedclip', 
                 preprocessing_method='none', category_ids=None, visualize=True, 
                 n_steps=4, evaluation_strategy='segmentation'):
        """
        Inicializa el pipeline.
        
        Parameters:
        -----------
        json_path : str
            Ruta al archivo COCO JSON.
        images_dir : str
            Directorio con las imágenes.
        model_name : str
            Nombre del modelo.
        preprocessing_method : str
            Método de preprocesamiento.
        category_ids : list
            IDs de categorías a incluir.
        visualize : bool
            Si mostrar visualizaciones.
        n_steps : int
            Número de pasos para crecer el árbol.
        evaluation_strategy : str
            Estrategia de evaluación: 'bbox' o 'segmentation'.
        """
        self.json_path = json_path
        self.images_dir = images_dir
        self.model_name = model_name
        self.preprocessing_method = preprocessing_method
        self.category_ids = category_ids or [4, 5]
        self.visualize = visualize
        self.n_steps = n_steps
        self.evaluation_strategy_name = evaluation_strategy
        
        print("⏳ Cargando modelo...")
        self.model = create_model(model_name)
        print(f"✅ Modelo cargado: {model_name}\n")
        
        # Variables de estado
        self.img = None
        self.boxes = None
        self.fname = None
        self.H = None
        self.W = None
        self.scale = 1.0  # Factor de escala aplicado a la imagen
        self.patch_data = None
        self.tree = None
        self.coco_data = None
        self.image_id = None
        self.evaluation_strategy = None
        self.ground_truth = None
    
    def load_image(self, image_index):
        """Carga una imagen del dataset."""
        print(f"📂 Explorando dataset...")
        
        # Cargar COCO data como diccionario JSON (no como objeto COCO)
        import json
        with open(self.json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        images_list = list_coco_images(self.json_path)
        print(f"   Total de imágenes: {len(images_list)}")
        
        if image_index >= len(images_list):
            raise IndexError(f"Índice {image_index} fuera de rango (máx: {len(images_list)-1})")
        
        print(f"\n📷 Cargando imagen {image_index}...")
        self.image_id = images_list[image_index]['id']
        
        self.img, self.boxes, self.fname = load_image_and_boxes_from_coco(
            self.json_path,
            self.images_dir,
            image_id=self.image_id,
            block_size=224,
            category_ids=self.category_ids,
            coco_data=self.coco_data
        )
        
        # Resize a tamaño estándar
        TARGET_SIZE = (1376, 1020)
        original_shape = self.img.shape[:2]
        scale_x = TARGET_SIZE[0] / self.img.shape[1]
        scale_y = TARGET_SIZE[1] / self.img.shape[0]
        scale = min(scale_x, scale_y)
        
        new_w = int(self.img.shape[1] * scale)
        new_h = int(self.img.shape[0] * scale)
        
        img_resized = cv2.resize(self.img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convertir a escala de grises
        if img_resized.ndim == 3:
            self.img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            self.img = img_resized
        
        # Aplicar preprocesamiento
        if self.preprocessing_method != 'none':
            self.img = apply_preprocessing(self.img, method=self.preprocessing_method)
        
        # Escalar boxes
        boxes_resized = []
        for cls, x1, y1, x2, y2 in self.boxes:
            x1_new = int(x1 * scale)
            y1_new = int(y1 * scale)
            x2_new = int(x2 * scale)
            y2_new = int(y2 * scale)
            boxes_resized.append((cls, x1_new, y1_new, x2_new, y2_new))
        
        self.boxes = boxes_resized
        self.H, self.W = self.img.shape[:2]
        self.scale = scale  # Guardar escala para usar en estrategias de evaluación
        
        print(f"✅ {self.fname}")
        print(f"   Tamaño original: {original_shape[1]}x{original_shape[0]} px")
        print(f"   Tamaño final: {self.W}x{self.H} px (escala: {scale:.2f})")
        print(f"   Anotaciones (núcleos): {len(self.boxes)}")
        print(f"   Preprocesamiento: {self.preprocessing_method.upper()}")
    
    def extract_embeddings(self):
        """Extrae embeddings de patches de la imagen."""
        print("\n⏳ Extrayendo embeddings...")
        
        try:
            if isinstance(self.model, BaseEmbeddingModel):
                self.patch_data = get_all_patch_embeddings_from_image(self.img, self.model)
            else:
                self.patch_data = get_all_patch_embeddings_from_image(
                    self.img, self.model, preprocess=None, tile_size=224
                )
        except Exception as e:
            raise RuntimeError(f"Error extrayendo embeddings: {e}")
        
        print(f"✅ {len(self.patch_data)} patches extraídos")
    
    def build_tree(self):
        """Construye el árbol de clustering."""
        print("\n" + "="*70)
        print("CONSTRUCCIÓN DEL ÁRBOL DE CLUSTERING")
        print("="*70)
        
        # Crear estrategia de evaluación
        from utils.evaluation_strategies import create_evaluation_strategy
        
        print(f"\n🔍 Configurando estrategia de evaluación: {self.evaluation_strategy_name}")
        self.evaluation_strategy = create_evaluation_strategy(
            self.evaluation_strategy_name
        )
        
        # Cargar ground truth CON ESCALA APLICADA
        # Pasar escala para que las máscaras se redimensionen correctamente
        self.ground_truth = self.evaluation_strategy.load_ground_truth(
            self.coco_data,
            self.image_id,
            self.category_ids,
            (self.H, self.W),
            scale=self.scale  # Factor de escala aplicado a la imagen
        )
        print(f"✅ Ground truth cargado\n")
        
        # Inicializar árbol
        self.tree = ClusteringTree(
            self.patch_data, 
            self.img, 
            self.ground_truth, 
            self.H, 
            self.W,
            self.evaluation_strategy
        )
        
        # Paso 0: División inicial (Fondo vs Tejido)
        self.tree.grow_step(visualize=self.visualize)
        
        # Pasos 1+: Crecer el árbol
        for step in range(1, self.n_steps):
            self.tree.grow_step(visualize=self.visualize)
            
            # Parar si no hay más hojas para dividir
            leaves = self.tree.root.get_all_leaves()
            if all(l.n_patches < 2 for l in leaves):
                print(f"\n⚠️ No hay más ramas para dividir. Parando en paso {step}.")
                break
        
        # Mostrar resumen
        self.tree.print_summary()
        
        return self.tree
    
    def run(self, image_index):
        """Ejecuta el pipeline completo."""
        print("\n" + "="*70)
        print("PIPELINE EXPLORADOR - ÁRBOL DE CLUSTERING")
        print("="*70)
        
        try:
            self.load_image(image_index)
            self.extract_embeddings()
            self.build_tree()
            
            print("\n" + "="*70)
            print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
            print("="*70 + "\n")
            
            return self.tree
        
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None


def generate_html_report(json_path, images_dir, output_path, n_steps=4, model_name='biomedclip', 
                        preprocessing_method='clahe', category_ids=None, evaluation_strategy='segmentation'):
    """
    Genera un reporte HTML procesando todas las imágenes del dataset.
    
    Parameters:
    -----------
    json_path : str
        Ruta al archivo COCO JSON
    images_dir : str
        Directorio con imágenes
    output_path : str
        Archivo de salida HTML
    n_steps : int
        Número de pasos del árbol
    model_name : str
        Nombre del modelo
    preprocessing_method : str
        Método de preprocesamiento
    category_ids : list
        IDs de categorías
    evaluation_strategy : str
        Estrategia de evaluación: 'bbox' o 'segmentation'
    """
    import base64
    import io
    from datetime import datetime
    
    if category_ids is None:
        category_ids = [4, 5]
    
    # Obtener lista de imágenes
    from utils.coco_loader import list_coco_images
    images_list = list_coco_images(json_path)
    
    print(f"\n📂 Total de imágenes en dataset: {len(images_list)}")
    print(f"⏳ Procesando {len(images_list)} imágenes...")
    print("="*70)
    
    # Crear pipeline
    pipeline = ImageProcessingPipeline(
        json_path=json_path,
        images_dir=images_dir,
        model_name=model_name,
        preprocessing_method=preprocessing_method,
        category_ids=category_ids,
        visualize=False,
        n_steps=n_steps,
        evaluation_strategy=evaluation_strategy
    )
    
    # Procesar imágenes
    results = []
    for i in range(len(images_list)):
        print(f"\n[{i+1}/{len(images_list)}] Procesando imagen...")
        
        try:
            tree = pipeline.run(i)
            
            if tree is None:
                print(f"      ⚠️ Error: tree es None")
                results.append(None)
                continue
            
            best_node = tree.get_best_node()
            optimal_steps = tree.get_optimal_steps()
            leaves = tree.root.get_all_leaves()
            leaves_with_metrics = [l for l in leaves if l.metrics is not None]
            
            if not leaves_with_metrics:
                print(f"      ⚠️ No hay hojas con métricas")
                results.append(None)
                continue
            
            best_leaf = max(leaves_with_metrics, key=lambda x: x.metrics['f1_coverage'])
            
            # Crear visualización
            fig, ax = plt.subplots(figsize=(14, 12))
            
            if pipeline.img.ndim == 2:
                ax.imshow(pipeline.img, cmap='gray')
            else:
                ax.imshow(pipeline.img)
            
            # Obtener datos de visualización de la estrategia
            viz_data = pipeline.evaluation_strategy.get_visualization_data(
                best_node.patches, 
                pipeline.img, 
                pipeline.ground_truth
            )
            
            # Renderizar según tipo de visualización
            if viz_data['type'] == 'bbox':
                # Estrategia de bounding boxes
                grupos = viz_data['groups']
                
                # Dibujar grupos
                for i_g, grupo in enumerate(grupos):
                    x1, y1, x2, y2 = grupo['position']
                    is_tp = best_node.metrics['pred_hits'][i_g] if i_g < len(best_node.metrics['pred_hits']) else False
                    color = (0.2, 0.8, 0.2) if is_tp else (0.8, 0.2, 0.2)
                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
                    ax.add_patch(rect)
                
                # Dibujar GT boxes
                boxes = viz_data['gt_boxes']
                gt_covered = viz_data['gt_covered']
                for gt_idx, (cls, x1, y1, x2, y2) in enumerate(boxes):
                    color = 'blue' if gt_covered[gt_idx] else 'orange'
                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.5, edgecolor=color, facecolor='none', linestyle='--')
                    ax.add_patch(rect)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    ax.plot(cx, cy, 'o', markersize=10, color=color, markeredgecolor='yellow', markeredgewidth=2)
                
                # Leyenda
                legend_elements = [
                    Rectangle((0, 0), 1, 1, fc=(0.2, 0.8, 0.2), alpha=0.5, label='TP (Correcto)'),
                    Rectangle((0, 0), 1, 1, fc=(0.8, 0.2, 0.2), alpha=0.5, label='FP (Incorrecto)'),
                    Rectangle((0, 0), 1, 1, fc='none', edgecolor='blue', linewidth=2, linestyle='--', label='GT cubierto'),
                    Rectangle((0, 0), 1, 1, fc='none', edgecolor='orange', linewidth=2, linestyle='--', label='GT no cubierto'),
                ]
                
                title = (f"Mejor Nodo: {best_node.node_id} (Paso {best_node.step})\\n"
                        f"TP={best_node.metrics['groups_TP']} | FP={best_node.metrics['groups_FP']} | "
                        f"F1={best_node.metrics['f1_coverage']:.3f}")
            
            elif viz_data['type'] == 'segmentation':
                # Estrategia de segmentación
                overlay = np.zeros((*pipeline.img.shape[:2], 3), dtype=np.uint8)
                
                tp_mask = viz_data['tp_mask']
                fp_mask = viz_data['fp_mask']
                fn_mask = viz_data['fn_mask']
                
                overlay[tp_mask > 0] = [0, 255, 0]
                overlay[fp_mask > 0] = [255, 0, 0]
                overlay[fn_mask > 0] = [0, 0, 255]
                
                ax.imshow(overlay, alpha=0.4)
                
                # Contornos de GT
                gt_masks = viz_data['gt_masks']
                for gt_mask in gt_masks:
                    contours = cv2.findContours(
                        gt_mask.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )[0]
                    for contour in contours:
                        contour = contour.squeeze()
                        if contour.ndim == 2:
                            ax.plot(contour[:, 0], contour[:, 1], 
                                   color='yellow', linewidth=2, linestyle='--')
                
                legend_elements = [
                    Rectangle((0, 0), 1, 1, fc=(0, 1, 0), alpha=0.5, label='TP'),
                    Rectangle((0, 0), 1, 1, fc=(1, 0, 0), alpha=0.5, label='FP'),
                    Rectangle((0, 0), 1, 1, fc=(0, 0, 1), alpha=0.5, label='FN'),
                    Rectangle((0, 0), 1, 1, fc='none', edgecolor='yellow', linewidth=2, linestyle='--', label='GT mask'),
                ]
                
                dice = best_node.metrics.get('dice', 0.0)
                iou = best_node.metrics.get('iou', 0.0)
                title = (f"Mejor Nodo: {best_node.node_id} (Paso {best_node.step})\\n"
                        f"DICE={dice:.3f} | IoU={iou:.3f} | F1={best_node.metrics['f1_coverage']:.3f}")
            
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.axis('off')
            
            # Convertir a base64
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            image_b64 = f"data:image/png;base64,{image_base64}"
            
            result = {
                'image_index': i,
                'filename': pipeline.fname,
                'dimensions': f"{pipeline.W}x{pipeline.H}",
                'gt_count': (len(pipeline.ground_truth) if isinstance(pipeline.ground_truth, list) 
                            else pipeline.ground_truth.get('n_instances', 1) 
                            if isinstance(pipeline.ground_truth, dict) 
                            else 1),
                'best_node': best_node.node_id,
                'optimal_steps': optimal_steps,
                'best_node_f1': best_node.metrics['f1_coverage'],
                'best_node_tp': best_node.metrics.get('groups_TP', best_node.metrics.get('tp_pixels', 0)),
                'best_node_fp': best_node.metrics.get('groups_FP', best_node.metrics.get('fp_pixels', 0)),
                'best_node_precision': best_node.metrics.get('group_precision', best_node.metrics.get('precision', 0)),
                'best_node_recall': best_node.metrics.get('gt_recall_coverage', best_node.metrics.get('recall', 0)),
                'best_node_dice': best_node.metrics.get('dice', 0),
                'best_node_iou': best_node.metrics.get('iou', 0),
                'best_leaf': best_leaf.node_id,
                'best_leaf_step': best_leaf.step,
                'best_leaf_f1': best_leaf.metrics['f1_coverage'],
                'visualization': image_b64,
                'step_stats': tree.get_metrics_by_step()
            }
            
            results.append(result)
            print(f"      ✅ F1: {best_node.metrics['f1_coverage']:.3f} | Pasos óptimos: {optimal_steps}")
        
        except Exception as e:
            print(f"      ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append(None)
    
    # Generar HTML
    print("\n" + "="*70)
    print("📄 Generando HTML...")
    
    successful = [r for r in results if r is not None]
    
    if successful:
        avg_f1 = np.mean([r['best_node_f1'] for r in successful])
        avg_precision = np.mean([r['best_node_precision'] for r in successful])
        avg_recall = np.mean([r['best_node_recall'] for r in successful])
        total_tp = sum(r['best_node_tp'] for r in successful)
        total_fp = sum(r['best_node_fp'] for r in successful)
        total_gt = sum(r['gt_count'] for r in successful)
        images_needing_fewer_steps = sum(1 for r in successful if r['optimal_steps'] < n_steps)
    else:
        avg_f1 = avg_precision = avg_recall = 0
        total_tp = total_fp = total_gt = 0
        images_needing_fewer_steps = 0
    
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Clustering</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 10px 40px rgba(0,0,0,0.3); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .config {{ background: #f5f5f5; padding: 20px; border-bottom: 1px solid #e0e0e0; }}
        .config h2 {{ font-size: 1.3em; margin-bottom: 15px; color: #333; }}
        .config-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .config-item {{ background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #667eea; }}
        .config-item strong {{ color: #667eea; display: block; margin-bottom: 5px; }}
        .summary {{ padding: 40px; background: white; }}
        .summary h2 {{ font-size: 1.8em; margin-bottom: 30px; color: #333; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 10px; text-align: center; box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3); }}
        .stat-card .value {{ font-size: 2.5em; font-weight: bold; margin-bottom: 10px; }}
        .stat-card .label {{ font-size: 0.95em; opacity: 0.9; }}
        .insights {{ background: #e8f4f8; padding: 25px; border-left: 4px solid #0288d1; border-radius: 5px; margin-bottom: 30px; }}
        .insights h3 {{ color: #0288d1; margin-bottom: 15px; font-size: 1.2em; }}
        .insights ul {{ list-style: none; padding-left: 0; }}
        .insights li {{ padding: 8px 0; border-bottom: 1px solid rgba(2, 136, 209, 0.2); }}
        .insights li:before {{ content: "✓ "; color: #4caf50; font-weight: bold; margin-right: 10px; }}
        .table-responsive {{ overflow-x: auto; margin-bottom: 40px; }}
        table {{ width: 100%; border-collapse: collapse; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        thead {{ background: #667eea; color: white; }}
        th {{ padding: 15px; text-align: left; font-weight: 600; font-size: 0.95em; }}
        td {{ padding: 12px 15px; border-bottom: 1px solid #e0e0e0; }}
        tbody tr:hover {{ background: #f5f5f5; }}
        .image-card {{ background: white; border: 1px solid #e0e0e0; border-radius: 10px; margin-bottom: 40px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .image-card-header {{ background: #f5f5f5; padding: 20px; border-bottom: 1px solid #e0e0e0; }}
        .image-card-header h3 {{ font-size: 1.3em; color: #333; margin-bottom: 10px; }}
        .image-card-body {{ padding: 20px; }}
        .image-card img {{ width: 100%; max-width: 1000px; border-radius: 5px; margin-bottom: 20px; }}
        .metric-box {{ background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 3px solid #667eea; display: inline-block; margin-right: 15px; margin-bottom: 10px; }}
        .metric-box strong {{ display: block; color: #667eea; margin-bottom: 5px; }}
        .metric-box span {{ font-size: 1.3em; font-weight: bold; color: #333; }}
        .footer {{ background: #f5f5f5; padding: 30px; text-align: center; color: #666; border-top: 1px solid #e0e0e0; }}
        .footer p {{ margin-bottom: 8px; }}
        .timestamp {{ font-size: 0.9em; color: #999; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Reporte del Pipeline de Clustering Adaptativo</h1>
            <p>Análisis de detección de núcleos celulares</p>
        </div>
        
        <div class="config">
            <h2>⚙️ Configuración</h2>
            <div class="config-grid">
                <div class="config-item"><strong>Modelo</strong>{model_name}</div>
                <div class="config-item"><strong>Preprocesamiento</strong>{preprocessing_method.upper()}</div>
                <div class="config-item"><strong>Pasos Máximos</strong>{n_steps}</div>
                <div class="config-item"><strong>Total de Imágenes</strong>{len(results)}</div>
            </div>
        </div>
        
        <div class="summary">
            <h2>📊 Resumen General</h2>
            <div class="stats-grid">
                <div class="stat-card"><div class="value">{len(successful)}</div><div class="label">Imágenes Procesadas</div></div>
                <div class="stat-card"><div class="value">{avg_f1:.3f}</div><div class="label">F1 Promedio</div></div>
                <div class="stat-card"><div class="value">{avg_precision:.3f}</div><div class="label">Precisión Promedio</div></div>
                <div class="stat-card"><div class="value">{avg_recall:.3f}</div><div class="label">Recall Promedio</div></div>
                <div class="stat-card"><div class="value">{total_tp}/{total_gt}</div><div class="label">TP / GT Total</div></div>
                <div class="stat-card"><div class="value">{total_fp}</div><div class="label">FP Total</div></div>
            </div>
            
            <div class="insights">
                <h3>💡 Insights Principales</h3>
                <ul>
                    <li><strong>Desempeño General:</strong> Detecta {(total_tp/total_gt*100):.1f}% de los núcleos (TP: {total_tp}/{total_gt})</li>
                    <li><strong>Falsos positivos:</strong> {total_fp} detecciones incorrectas en total</li>
                    <li><strong>Precisión:</strong> Promedio {avg_precision:.1%} de las detecciones son correctas</li>
                    <li><strong>Recall:</strong> Promedio {avg_recall:.1%} de los núcleos son encontrados</li>
                </ul>
            </div>
        </div>
        
        <div class="summary">
            <h2>🖼️ Análisis Detallado por Imagen</h2>
"""
    
    for result in successful:
        html += f"""
            <div class="image-card">
                <div class="image-card-header">
                    <h3>{result['filename']}</h3>
                </div>
                <div class="image-card-body">
                    <img src="{result['visualization']}" alt="{result['filename']}">
                    <div>
                        <div class="metric-box"><strong>Pasos Óptimos</strong><span>{result['optimal_steps']}/{n_steps}</span></div>
                        <div class="metric-box"><strong>TP</strong><span>{result['best_node_tp']}</span></div>
                        <div class="metric-box"><strong>FP</strong><span>{result['best_node_fp']}</span></div>
                        <div class="metric-box"><strong>Precisión</strong><span>{result['best_node_precision']:.3f}</span></div>
                        <div class="metric-box"><strong>Recall</strong><span>{result['best_node_recall']:.3f}</span></div>
                        <div class="metric-box"><strong>F1</strong><span>{result['best_node_f1']:.3f}</span></div>
                    </div>
                </div>
            </div>
"""
    
    html += f"""
        </div>
        
        <div class="footer">
            <p><strong>Reporte generado automáticamente</strong></p>
            <p class="timestamp">Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Reporte guardado en: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline explorador con árbol de clustering adaptativo"
    )
    
    parser.add_argument(
        '--json', type=str, required=True,
        help='Ruta al archivo COCO JSON con anotaciones'
    )
    parser.add_argument(
        '--images_dir', type=str, required=True,
        help='Directorio con las imágenes'
    )
    parser.add_argument(
        '--index', type=int, default=0,
        help='Índice de la imagen a procesar (por defecto: 0)'
    )
    parser.add_argument(
        '--n_steps', type=int, default=4,
        help='Número de pasos para crecer el árbol (por defecto: 4)'
    )
    parser.add_argument(
        '--model', type=str, default='biomedclip',
        choices=['biomedclip', 'uni', 'optimus', 'uni2'],
        help='Modelo de embeddings a usar (por defecto: biomedclip)'
    )
    parser.add_argument(
        '--preprocessing', type=str, default='none',
        choices=['clahe', 'equalize', 'normalize', 'none'],
        help='Método de preprocesamiento (por defecto: none)'
    )
    parser.add_argument(
        '--visualize', action='store_true', default=True,
        help='Mostrar visualizaciones durante el procesamiento'
    )
    parser.add_argument(
        '--no-visualize', dest='visualize', action='store_false',
        help='No mostrar visualizaciones'
    )
    
    args = parser.parse_args()
    
    # Validar rutas
    if not os.path.exists(args.json):
        print(f"❌ Error: Archivo JSON no encontrado: {args.json}")
        sys.exit(1)
    
    if not os.path.isdir(args.images_dir):
        print(f"❌ Error: Directorio no encontrado: {args.images_dir}")
        sys.exit(1)
    
    # Crear y ejecutar pipeline
    pipeline = ImageProcessingPipeline(
        json_path=args.json,
        images_dir=args.images_dir,
        model_name=args.model,
        preprocessing_method=args.preprocessing,
        visualize=args.visualize,
        n_steps=args.n_steps
    )
    
    tree = pipeline.run(args.index)
    
    if tree:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
