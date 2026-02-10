"""
Configuración compartida del proyecto.
Centraliza todos los parámetros en un solo lugar.
"""

class BaseConfig:
    """Configuración base."""
    
    # ===== MODELO FUNDACIONAL =====
    model_name = 'biomedclip'  # 'biomedclip', 'uni', 'optimus', 'uni2'
    model_device = None  # None = auto-detect (cuda si disponible)
    
    # ===== DATOS =====
    json_path = '/kaggle/input/cric-dataset/classifications.json'
    base_path = '/kaggle/input/cric-dataset'
    
    # ===== PARÁMETROS DE IMAGEN =====
    box_size = 224              # Tamaño de tile para BiomedCLIP (ViT-B: 16x16 patches → 224x224)
    boxes_size_gt = 60          # Tamaño de box GT alrededor del centro del núcleo
    
    # ===== CLUSTERING =====
    clustering_method = 'kmeans'  # 'kmeans', 'agglomerative', 'dbscan'
    
    # ===== LIMPIEZA =====
    min_patches_componente = 3      # Mínimo de patches conexos por componente
    dilate_px_componentes = 32      # Dilatación para cerrar gaps pequeños
    connectivity = 8                # 4 u 8 vecinos
    
    # ===== EVALUACIÓN =====
    match_mode = 'cover_gt'        # 'center', 'iou', 'cover_gt', etc.
    cover_gt_thr = 0.20            # Umbral de cobertura del GT para match
    iou_thr = 0.5
    
    # ===== OUTPUT =====
    out_root = './eval_multistep'
    show_in_notebook = False
    save_figs = True
    save_pickle = False
    dpi_figs = 120


class KaggleConfig(BaseConfig):
    """Configuración para Kaggle Notebooks."""
    json_path = '/kaggle/input/cric-dataset/classifications.json'
    base_path = '/kaggle/input/cric-dataset'
    out_root = '/kaggle/working/eval_multistep'


class LocalConfig(BaseConfig):
    """Configuración para ejecución local."""
    json_path = './data/classifications.json'
    base_path = './data/images'
    out_root = './results'
    show_in_notebook = True


class DevelopConfig(BaseConfig):
    """Configuración para desarrollo (datos pequeños, debug=True)."""
    box_size = 224
    boxes_size_gt = 60
    min_patches_componente = 1  # Más permisivo
    images_to_analyze = [150, 151]  # Solo 2 imágenes
    show_in_notebook = True
    save_figs = True


# ===== BETHESDA CLASSIFICATION SYSTEM =====
BETHESDA_CLASSES = ['ASC-H', 'HSIL', 'LSIL', 'SCC', 'NILM', 'ASC-US']

BETHESDA_COLORS = {
    0: 'orange',   # ASC-H
    1: 'purple',   # HSIL
    2: 'blue',     # LSIL
    3: 'red',      # SCC
    4: 'green',    # NILM
    5: 'yellow'    # ASC-US
}
