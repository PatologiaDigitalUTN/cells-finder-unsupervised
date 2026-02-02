"""
Utilidades de visualización para debugging y análisis.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import numpy as np
from .evaluation import _to_xyxy, _center


def mostrar_imagen_con_boxes(img, boxes, title="Imagen con boxes"):
    """Dibuja imagen con bounding boxes anotados."""
    bethesda_colors = {
        0: 'orange',
        1: 'purple',
        2: 'blue',
        3: 'red',
        4: 'green',
        5: 'yellow'
    }
    
    fig, ax = plt.subplots(1, figsize=(12, 12))
    
    if img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    
    for idx, (cls, x1, y1, x2, y2) in enumerate(boxes):
        color = bethesda_colors.get(cls, 'gray')
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f"{idx}", color='white', fontsize=9, backgroundcolor=color)
    
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualizar_clusters_basicos(img, clustered_patches, boxes=None, cluster_field='cluster', 
                                 grid_size=224, alpha=0.4, title="Clusters"):
    """Visualiza clusters coloreados sobre la imagen."""
    unique_clusters = sorted(set(p.get(cluster_field) for p in clustered_patches if cluster_field in p))
    cluster_colors = {cid: cm.get_cmap('tab10')(i % 10) for i, cid in enumerate(unique_clusters)}

    bethesda_colors = {0: 'orange', 1: 'purple', 2: 'blue', 3: 'red', 4: 'green', 5: 'yellow'}

    fig, ax = plt.subplots(figsize=(12, 12))

    if img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)

    # Dibujar clusters
    for patch in clustered_patches:
        if 'position' not in patch or cluster_field not in patch:
            continue
        x1, y1, x2, y2 = patch['position']
        cluster = patch[cluster_field]
        color = cluster_colors.get(cluster)
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=1, edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(rect)

    # Dibujar boxes GT si existen
    if boxes:
        for (cls, x1, y1, x2, y2) in boxes:
            ax.plot((x1 + x2) // 2, (y1 + y2) // 2, 'o', markersize=6, color=bethesda_colors.get(cls))

    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualizar_limpieza_patches(img, kept, removed, boxes=None, alpha=0.4, 
                                 title="Limpieza de patches"):
    """Visualiza patches mantenidos (verde) y removidos (rojo)."""
    fig, ax = plt.subplots(figsize=(12, 12))

    if img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)

    ax.axis('off')
    ax.set_title(f"{title} | kept={len(kept)} removed={len(removed)}")

    # Removidos (rojo)
    for p in removed:
        x1, y1, x2, y2 = p['position']
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               ec=(0.9, 0.2, 0.2), fc=(0.9, 0.2, 0.2), alpha=alpha, lw=2))
    
    # Kept (verde)
    for p in kept:
        x1, y1, x2, y2 = p['position']
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               ec=(0.1, 0.9, 0.1), fc=(0.1, 0.9, 0.1), alpha=alpha, lw=2))

    # Boxes
    if boxes:
        for b in boxes:
            if isinstance(b, (list, tuple)):
                if len(b) == 5:
                    _, x1, y1, x2, y2 = b
                else:
                    x1, y1, x2, y2 = b[-4:]
            else:
                continue
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, ec='blue', lw=1.5))

    handles = [
        plt.Line2D([0], [0], marker='s', linestyle='None', markersize=10,
                   markerfacecolor=(0.1, 0.9, 0.1), markeredgecolor='k', label='kept'),
        plt.Line2D([0], [0], marker='s', linestyle='None', markersize=10,
                   markerfacecolor=(0.9, 0.2, 0.2), markeredgecolor='k', label='removed'),
    ]
    if boxes:
        handles.append(plt.Line2D([0], [0], color='blue', lw=1.5, label='GT'))
    
    ax.legend(handles=handles, loc='upper right')
    plt.tight_layout()
    plt.show()


def visualizar_grupos_vs_boxes(img, grupos, boxes_gt, match_mode='cover_gt', 
                                cover_gt_thr=0.3, alpha=0.35, figsize=(12, 12)):
    """Visualiza grupos (TP en verde, FP en rojo) vs GT boxes."""
    from .evaluation import _to_xyxy, _center, _overlap_area, _area, _inside, _iou

    H, W = img.shape[:2]
    pred_xyxy = [_to_xyxy(g['position'], (H, W)) for g in grupos]
    gt_xyxy = [_to_xyxy(b, (H, W)) for b in boxes_gt]

    def _cover_gt(a, b):
        inter = _overlap_area(a, b)
        agb = _area(b)
        return (inter / agb) if agb > 0 else 0.0

    def _match(p, g):
        c = _center(p)
        if match_mode == 'center':
            return (c[0] >= g[0]) and (c[0] <= g[2]) and (c[1] >= g[1]) and (c[1] <= g[3])
        elif match_mode == 'cover_gt':
            return _cover_gt(p, g) >= cover_gt_thr
        return False

    pred_hits = []
    gt_hit_counts = [0] * len(gt_xyxy)

    for p in pred_xyxy:
        hit = False
        for i, g in enumerate(gt_xyxy):
            if _match(p, g):
                hit = True
                gt_hit_counts[i] += 1
        pred_hits.append(hit)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    ax.axis('off')

    # GT
    for i, g in enumerate(gt_xyxy):
        color = 'lime' if gt_hit_counts[i] > 0 else 'orange'
        x1, y1, x2, y2 = g
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=1.8))

    # Grupos
    for j, p in enumerate(pred_xyxy):
        color = 'lime' if pred_hits[j] else 'red'
        x1, y1, x2, y2 = p
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, ec=color, fc=color, alpha=alpha, lw=2))

    TP = sum(pred_hits)
    FP = len(pred_hits) - TP
    GT_hit = sum(1 for c in gt_hit_counts if c > 0)
    GT_tot = len(gt_xyxy)

    ax.set_title(f"TP={TP} FP={FP} | GT_hit={GT_hit}/{GT_tot}")
    plt.tight_layout()
    plt.show()
