"""
Utilidades para evaluación y cálculo de métricas.
"""

import numpy as np
import cv2
from collections import Counter
from sklearn.metrics import confusion_matrix


def _to_xyxy(box, img_shape=None):
    """Convierte diferentes formatos de box a (x1, y1, x2, y2)."""
    def _maybe_denorm(v):
        if img_shape is None:
            return v
        v = np.asarray(v, dtype=float).ravel().copy()
        if v.size >= 4 and v.min() >= 0.0 and v.max() <= 1.0001:
            H, W = img_shape[:2]
            v[0] *= W
            v[1] *= H
            v[2] *= W
            v[3] *= H
        return v

    if isinstance(box, dict):
        if 'position' in box:
            x1, y1, x2, y2 = _maybe_denorm(box['position'])
            return float(x1), float(y1), float(x2), float(y2)
        if 'bbox' in box:
            x, y, w, h = _maybe_denorm(box['bbox'])
            return float(x), float(y), float(x + w), float(y + h)
        raise ValueError("Box dict sin claves conocidas")
    
    arr = _maybe_denorm(box)
    if arr[2] > arr[0] and arr[3] > arr[1]:  # xyxy
        x1, y1, x2, y2 = arr[:4]
        return float(x1), float(y1), float(x2), float(y2)
    else:  # xywh
        x, y, w, h = arr[:4]
        return float(x), float(y), float(x + w), float(y + h)


def _center(xyxy):
    """Centro de un box (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _inside(pt, xyxy):
    """¿El punto está dentro del box?"""
    x, y = pt
    x1, y1, x2, y2 = xyxy
    return (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2)


def _overlap_area(a, b):
    """Área de solapamiento entre dos boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return iw * ih


def _area(xyxy):
    """Área de un box."""
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _cover_gt(a, b):
    """Fracción del GT b cubierta por pred a."""
    inter = _overlap_area(a, b)
    agb = _area(b)
    return (inter / agb) if agb > 0 else 0.0


def _iou(a, b):
    """IoU entre dos boxes."""
    inter = _overlap_area(a, b)
    if inter <= 0:
        return 0.0
    area_a = _area(a)
    area_b = _area(b)
    union = max(area_a + area_b - inter, 1e-9)
    return inter / union


def evaluar_grupos_vs_boxes_plus(
    img,
    grupos,
    boxes_gt,
    match_mode='cover_gt',
    iou_thr=0.5,
    cover_gt_thr=0.3,
    cover_pred_thr=0.5
):
    """
    Evalúa grupos (patches agrupados) vs GT boxes.
    
    Parameters:
    -----------
    img : np.ndarray
        Imagen.
    grupos : list[dict]
        Grupos con 'position'=(x1,y1,x2,y2).
    boxes_gt : list
        GT boxes.
    match_mode : str
        'center', 'iou', 'cover_gt', etc.
    
    Returns:
    --------
    dict : Métricas (TP, FP, precision, recall, F1).
    """
    H, W = img.shape[:2]

    def _match(p, g, mode):
        c = _center(p)
        if mode == 'center':
            return _inside(c, g)
        elif mode == 'iou':
            return _iou(p, g) >= iou_thr
        elif mode == 'overlap':
            return _overlap_area(p, g) > 0.0
        elif mode == 'cover_gt':
            return _cover_gt(p, g) >= cover_gt_thr
        elif mode == 'cover_pred':
            return _cover_pred(p, g) >= cover_pred_thr
        return False

    pred_xyxy = [_to_xyxy(g['position'], (H, W)) for g in grupos]
    gt_xyxy = [_to_xyxy(b, (H, W)) for b in boxes_gt]

    pred_hits = []
    gt_hit_counts = [0] * len(gt_xyxy)

    for p in pred_xyxy:
        hit = False
        for i, g in enumerate(gt_xyxy):
            if _match(p, g, match_mode):
                hit = True
                gt_hit_counts[i] += 1
        pred_hits.append(hit)

    TPg = int(sum(pred_hits))
    FPg = int(len(pred_hits) - TPg)
    GThit = int(sum(1 for c in gt_hit_counts if c > 0))
    GTtot = int(len(gt_xyxy))

    group_precision = TPg / (TPg + FPg) if (TPg + FPg) > 0 else 0.0
    gt_recall_cov = GThit / GTtot if GTtot > 0 else 0.0
    f1_cov = (2 * group_precision * gt_recall_cov / (group_precision + gt_recall_cov)
              if (group_precision + gt_recall_cov) > 0 else 0.0)

    return {
        'groups_TP': TPg,
        'groups_FP': FPg,
        'gt_hit': GThit,
        'gt_total': GTtot,
        'group_precision': group_precision,
        'gt_recall_coverage': gt_recall_cov,
        'f1_coverage': f1_cov,
        'pred_hits': pred_hits,
        'gt_hit_counts': gt_hit_counts,
        'match_mode': match_mode,
    }


def limpiar_patches_por_componentes_mask(
    img,
    patches,
    min_patches=3,
    connectivity=8,
    dilate_px=0
):
    """
    Mantiene solo patches que pertenecen a componentes conexas con ≥min_patches.
    
    Parameters:
    -----------
    img : np.ndarray
        Imagen de referencia (para shape).
    patches : list[dict]
        Patches con 'position'=(x1,y1,x2,y2).
    min_patches : int
        Mínimo de patches por componente.
    connectivity : int
        4 u 8 vecinos.
    dilate_px : int
        Dilatación opcional para cerrar gaps.
    
    Returns:
    --------
    kept, removed, debug : Patches mantenidos, removidos, estadísticas.
    """
    H, W = img.shape[:2]

    if not patches:
        return [], [], {'num_components': 0}

    # Máscara binaria
    mask = np.zeros((H, W), dtype=np.uint8)
    rects = []

    for p in patches:
        x1, y1, x2, y2 = p['position']
        x1, x2 = int(max(0, min(W, np.floor(x1)))), int(max(0, min(W, np.ceil(x2))))
        y1, y2 = int(max(0, min(H, np.floor(y1)))), int(max(0, min(H, np.ceil(y2))))
        
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        rects.append((x1, y1, x2, y2))

    # Dilatación opcional
    if dilate_px and dilate_px > 0:
        k = int(dilate_px)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Componentes conexas
    num_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=connectivity)

    # Contar patches por componente
    label_counts = {}
    patch_labels = []

    for (x1, y1, x2, y2) in rects:
        cx = int((x1 + x2) // 2)
        cy = int((y1 + y2) // 2)
        cx = max(0, min(W - 1, cx))
        cy = max(0, min(H - 1, cy))
        
        lab = int(labels[cy, cx])
        patch_labels.append(lab)
        if lab != 0:
            label_counts[lab] = label_counts.get(lab, 0) + 1

    # Seleccionar componentes válidas
    valid_labels = {lab for lab, cnt in label_counts.items() if cnt >= min_patches}

    kept_idx = [i for i, lab in enumerate(patch_labels) if lab in valid_labels and lab != 0]
    removed_idx = [i for i in range(len(patches)) if i not in kept_idx]

    kept = [patches[i] for i in kept_idx]
    removed = [patches[i] for i in removed_idx]

    debug = {
        'num_components': num_labels - 1,
        'label_counts_patches': dict(sorted(label_counts.items(), key=lambda kv: -kv[1])),
        'kept_idx': kept_idx,
        'removed_idx': removed_idx,
    }

    return kept, removed, debug


def agrupar_patches_en_grupos(
    img,
    patches,
    min_patches_por_grupo=1,
    dilate_px=0,
    connectivity=8
):
    """
    Agrupa patches en componentes conexas, devolviendo grupos con bbox y estadísticas.
    
    Returns:
    --------
    groups : list[dict]
        Cada grupo tiene 'position', 'n_patches', 'area_px', 'centroid', 'patch_indices'.
    debug : dict
        Estadísticas de clustering.
    """
    H, W = img.shape[:2]

    if not patches:
        return [], {'num_components': 0}

    # Máscara
    mask = np.zeros((H, W), dtype=np.uint8)
    rects = []

    for p in patches:
        x1, y1, x2, y2 = p['position']
        x1, x2 = int(max(0, min(W, np.floor(x1)))), int(max(0, min(W, np.ceil(x2))))
        y1, y2 = int(max(0, min(H, np.floor(y1)))), int(max(0, min(H, np.ceil(y2))))
        
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        rects.append((x1, y1, x2, y2))

    if dilate_px and dilate_px > 0:
        k = int(dilate_px)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Componentes con stats
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=connectivity
    )

    # Contar patches
    label_counts = {}
    patch_labels = []

    for (x1, y1, x2, y2) in rects:
        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
        cx, cy = max(0, min(W - 1, cx)), max(0, min(H - 1, cy))
        lab = int(labels[cy, cx])
        patch_labels.append(lab)
        if lab != 0:
            label_counts[lab] = label_counts.get(lab, 0) + 1

    # Grupos válidos
    groups = []
    for lab, cnt in label_counts.items():
        if cnt < min_patches_por_grupo:
            continue
        
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        area = int(stats[lab, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[lab][0]), float(centroids[lab][1])

        patch_idx = [i for i, pl in enumerate(patch_labels) if pl == lab]

        groups.append({
            'position': (float(x), float(y), float(x + w), float(y + h)),
            'n_patches': int(cnt),
            'area_px': area,
            'centroid': (cx, cy),
            'label': int(lab),
            'patch_indices': patch_idx
        })

    debug = {
        'num_components': int(num - 1),
        'label_patch_counts': dict(sorted(label_counts.items(), key=lambda kv: -kv[1]))
    }

    return groups, debug
