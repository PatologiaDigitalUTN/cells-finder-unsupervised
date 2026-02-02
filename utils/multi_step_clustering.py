"""
Funciones para clustering multi-paso: fondo/tejido, núcleos/citoplasma, etc.
"""

import numpy as np
import cv2
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA


def run_block_clustering_on_embeddings(crops, method='kmeans', n_clusters=5):
    """
    Clustering sobre embeddings de patches.
    
    Parameters:
    -----------
    crops : list[dict]
        Patches con 'embedding' (puede ser torch.Tensor o np.ndarray).
    method : str
        'kmeans', 'agglomerative' o 'dbscan'.
    n_clusters : int
        Número de clusters (para kmeans y agglomerative).
    
    Returns:
    --------
    crops : list[dict]
        Crops con campo 'cluster' añadido.
    """
    valid_crops = [c for c in crops if c.get('embedding') is not None]
    
    if not valid_crops:
        return crops

    # Convertir embeddings a np.ndarray
    embeddings = []
    for c in valid_crops:
        emb = c['embedding']
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        elif not isinstance(emb, np.ndarray):
            emb = np.asarray(emb)
        embeddings.append(emb.ravel())
    
    X = np.vstack(embeddings)

    # Clustering
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity='cosine')
    elif method == 'dbscan':
        model = DBSCAN(eps=0.7, min_samples=3)
    else:
        raise ValueError(f"Método no soportado: {method}")

    cluster_labels = model.fit_predict(X)

    # Asignar clusters a crops
    for crop, cluster in zip(valid_crops, cluster_labels):
        crop['cluster'] = int(cluster)

    # Copiar para crops sin embedding (comparar por identidad de objeto)
    valid_crops_ids = {id(c) for c in valid_crops}
    for crop in crops:
        if id(crop) not in valid_crops_ids and 'cluster' not in crop:
            crop['cluster'] = -1  # Marcar como inválido

    return crops


def refinar_cluster_con_kmeans(
    patches,
    cluster_id,
    nuevo_k=3,
    cluster_field='cluster',
    new_field='subcluster'
):
    """
    Aplica KMeans k=nuevo_k a los patches de un cluster específico.
    
    Parameters:
    -----------
    patches : list[dict]
        Lista de patches con embeddings.
    cluster_id : int
        ID del cluster a refinar.
    nuevo_k : int
        Número de sub-clusters.
    cluster_field : str
        Campo que contiene el cluster actual.
    new_field : str
        Campo donde guardar el nuevo clustering.
    
    Returns:
    --------
    nuevos_patches : list[dict]
        Patches con el nuevo campo añadido.
    """
    sub_patches = [p for p in patches if p.get(cluster_field) == cluster_id and p.get('embedding') is not None]

    if len(sub_patches) < nuevo_k:
        print(f"⚠️ Solo {len(sub_patches)} patches en cluster {cluster_id}, necesito {nuevo_k}")
        return patches

    # Convertir embeddings
    X_list = []
    for p in sub_patches:
        emb = p['embedding']
        if isinstance(emb, torch.Tensor):
            arr = emb.detach().cpu().numpy()
        elif isinstance(emb, np.ndarray):
            arr = emb.copy()
        else:
            arr = np.asarray(emb)
        X_list.append(arr.ravel())
    
    X = np.vstack(X_list)

    # KMeans
    kmeans = KMeans(n_clusters=nuevo_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Resultado
    nuevos_patches = []
    for p, sublabel in zip(sub_patches, labels):
        q = p.copy()
        q[new_field] = int(sublabel)
        nuevos_patches.append(q)
    
    # Copiar patches que no se refinaron
    for p in patches:
        if p not in sub_patches:
            nuevos_patches.append(p)

    return nuevos_patches


def _to_bgr(img):
    """Convierte imagen a BGR para procesamiento."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Formato no soportado: {img.shape}")


def _colorfulness_bgr(bgr):
    """
    Calcula colorfulness (Hasler & Suesstrunk).
    Valores más altos = más color.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)


def _roi_from_patch(img, p, expand=0.0):
    """Extrae ROI de un patch con expansión opcional."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = p['position']
    
    if expand and expand > 0:
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        w, h = (x2 - x1), (y2 - y1)
        w2, h2 = w * (1.0 + expand), h * (1.0 + expand)
        x1 = cx - w2 / 2
        x2 = cx + w2 / 2
        y1 = cy - h2 / 2
        y2 = cy + h2 / 2
    
    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(W, int(np.ceil(x2)))
    y2 = min(H, int(np.ceil(y2)))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return img[y1:y2, x1:x2]


def _cluster_metrics(img, patches, cluster_field, cid, use_context=False, expand=0.5):
    """
    Calcula métricas promedio de brillo y colorfulness para un cluster.
    """
    sub = [p for p in patches if p.get(cluster_field) == cid]
    if not sub:
        return dict(count=0, gray_mean=np.inf, color_mean=np.inf)
    
    gray_vals, color_vals = [], []
    base = _to_bgr(img)
    
    for p in sub:
        roi = _roi_from_patch(base, p, expand=(expand if use_context else 0.0))
        if roi is None:
            continue
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_vals.append(float(gray.mean()))
        color_vals.append(float(_colorfulness_bgr(roi)))
    
    if len(gray_vals) == 0:
        return dict(count=0, gray_mean=np.inf, color_mean=np.inf)
    
    return dict(
        count=len(gray_vals),
        gray_mean=float(np.mean(gray_vals)),
        color_mean=float(np.mean(color_vals))
    )


def decidir_fondo_vs_tejido(img, patches, cluster_field='cluster'):
    """
    Decide cuál cluster es fondo y cuál es tejido basado en brillo y color.
    Fondo: más brillante y menos colorido.
    """
    cids = sorted(set(p.get(cluster_field) for p in patches if cluster_field in p))
    if len(cids) != 2:
        raise ValueError(f"Se esperaban 2 clusters, hay {len(cids)}")
    
    s = {cid: _cluster_metrics(img, patches, cluster_field, cid, use_context=False) for cid in cids}
    scores = {cid: (s[cid]['gray_mean'] - s[cid]['color_mean']) for cid in cids}
    
    fondo = max(scores, key=scores.get)
    tejido = [c for c in cids if c != fondo][0]
    
    print(f"[Paso 1] fondo={fondo} (gray={s[fondo]['gray_mean']:.1f}, color={s[fondo]['color_mean']:.1f}) | "
          f"tejido={tejido} (gray={s[tejido]['gray_mean']:.1f}, color={s[tejido]['color_mean']:.1f})")
    
    return fondo, tejido, s


def decidir_nucleos_vs_citoplasma(img, patches, cluster_field='subcluster'):
    """
    Decide cuál cluster contiene núcleos y cuál citoplasma.
    Núcleos: más oscuros.
    """
    cids = sorted(set(p.get(cluster_field) for p in patches if cluster_field in p))
    if len(cids) != 2:
        raise ValueError(f"Se esperaban 2 clusters, hay {len(cids)}")
    
    s = {cid: _cluster_metrics(img, patches, cluster_field, cid, use_context=False) for cid in cids}
    nucleos = min(cids, key=lambda c: s[c]['gray_mean'])
    cytos = [c for c in cids if c != nucleos][0]
    
    print(f"[Paso 2] nucleos={nucleos} (gray={s[nucleos]['gray_mean']:.1f}) | "
          f"citoplasma={cytos} (gray={s[cytos]['gray_mean']:.1f})")
    
    return nucleos, cytos, s
