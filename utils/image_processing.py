"""
Utilidades para cargar imágenes, extraer bounding boxes y procesamiento básico.
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path


BETHESDA_CLASSES = ['ASC-H', 'HSIL', 'LSIL', 'SCC', 'NILM', 'ASC-US']


def normalize_class(bethesda_label):
    """Normaliza etiquetas Bethesda."""
    if bethesda_label == 'Negative for intraepithelial lesion':
        return 'NILM'
    return bethesda_label


def load_image_and_boxes_from_json(json_path, base_path, index=0, convertToGrayScale=False, boxes_size=90):
    """
    Carga imagen y boxes desde JSON.
    
    Parameters:
    -----------
    json_path : str
        Ruta al archivo JSON con anotaciones.
    base_path : str
        Ruta base de las imágenes.
    index : int
        Índice de la imagen en el JSON.
    convertToGrayScale : bool
        Si convertir a escala de grises.
    boxes_size : int
        Tamaño de cada box alrededor del centro del núcleo.
    
    Returns:
    --------
    img : np.ndarray
        Imagen cargada.
    boxes : list
        Lista de (class, x1, y1, x2, y2).
    image_file : str
        Nombre del archivo de imagen.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_entry = data[index]
    image_file = image_entry['image_name']
    image_path = os.path.join(base_path, image_file)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if convertToGrayScale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape
    else:
        h, w, _ = img.shape

    boxes = []
    for cell in image_entry['classifications']:
        cls_name = normalize_class(cell['bethesda_system'])
        if cls_name not in BETHESDA_CLASSES:
            continue
        cls = BETHESDA_CLASSES.index(cls_name)
        cx, cy = cell['nucleus_x'], cell['nucleus_y']

        half = boxes_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        if x2 > x1 and y2 > y1:
            boxes.append((cls, x1, y1, x2, y2))

    return img, boxes, image_file


def load_image_and_boxes_from_json_cropped(
    json_path,
    base_path,
    index=0,
    convertToGrayScale=False,
    boxes_size=90,
    block_size=224
):
    """
    Carga imagen y boxes, recortando la imagen a múltiplos de block_size.
    Útil para trabajar con tiles que encajen en grillas.
    
    Parameters:
    -----------
    json_path : str
        Ruta al archivo JSON.
    base_path : str
        Ruta base de imágenes.
    index : int
        Índice de la imagen.
    convertToGrayScale : bool
        Convertir a gris.
    boxes_size : int
        Tamaño del box alrededor del centro.
    block_size : int
        Tamaño de bloque para recorte (imagen será múltiplo de esto).
    
    Returns:
    --------
    img_out : np.ndarray
        Imagen recortada.
    boxes : list
        Boxes ajustados a la imagen recortada.
    image_file : str
        Nombre del archivo.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_entry = data[index]
    image_file = image_entry['image_name']
    image_path = os.path.join(base_path, image_file)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Recorte a múltiplos de block_size
    if block_size is None or block_size <= 0:
        raise ValueError("block_size debe ser positivo.")

    H, W = img_rgb.shape[:2]
    Hc = (H // block_size) * block_size
    Wc = (W // block_size) * block_size
    if Hc == 0 or Wc == 0:
        raise ValueError(f"Imagen {H}x{W} es muy chica para block_size={block_size}.")

    img_crop = img_rgb[:Hc, :Wc]

    # Convertir a gris si necesario
    if convertToGrayScale:
        img_out = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
        h, w = img_out.shape
    else:
        img_out = img_crop
        h, w, _ = img_out.shape

    # Construir boxes
    half = int(boxes_size) // 2
    boxes = []
    for cell in image_entry['classifications']:
        cls_name = normalize_class(cell['bethesda_system'])
        if cls_name not in BETHESDA_CLASSES:
            continue
        cls = BETHESDA_CLASSES.index(cls_name)
        cx, cy = int(cell['nucleus_x']), int(cell['nucleus_y'])

        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        if x2 > x1 and y2 > y1:
            boxes.append((cls, x1, y1, x2, y2))

    return img_out, boxes, image_file


def extract_bounding_boxes(img, boxes):
    """Extrae recortes (crops) basados en boxes."""
    crops = []
    for idx, (cls, x1, y1, x2, y2) in enumerate(boxes):
        crop = img[y1:y2, x1:x2]
        crops.append({
            'class': cls,
            'image': crop,
            'index': idx
        })
    return crops


def tile_image(img, tile_size=224, stride=224):
    """
    Divide la imagen en tiles de tamaño tile_size x tile_size.
    
    Returns:
    --------
    tiles : list
        Lista de tiles (np.ndarray).
    positions : list
        Lista de (x1, y1, x2, y2) para cada tile.
    """
    h, w = img.shape[:2]
    tiles = []
    positions = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = img[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((x, y, x + tile_size, y + tile_size))

    return tiles, positions


def apply_preprocessing(image, method='clahe'):
    """
    Aplica preprocesamiento para mejorar contraste y claridad de la imagen.
    
    Métodos disponibles:
    - 'clahe': CLAHE (Contrast Limited Adaptive Histogram Equalization) - RECOMENDADO
      para imágenes médicas. Mejora contraste localmente sin artefactos.
    - 'equalize': Equalización de histograma estándar. Más agresivo.
    - 'normalize': Normalización por desviación estándar. Útil para variaciones globales de iluminación.
    - 'none': Sin preprocesamiento.
    
    Parameters:
    -----------
    image : np.ndarray
        Imagen en escala de grises.
    method : str
        Método de preprocesamiento a aplicar.
    
    Returns:
    --------
    np.ndarray
        Imagen procesada.
    """
    if method == 'none' or image is None:
        return image
    
    # Asegurar que es escala de grises
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'clahe':
        # CLAHE: mejor para imágenes médicas, mantiene detalles locales
        # clipLimit controla el contraste (mayor = más contraste)
        # tileGridSize define el tamaño de los tiles adaptativos
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    elif method == 'equalize':
        # Equalización estándar (más agresivo que CLAHE)
        return cv2.equalizeHist(image)
    
    elif method == 'normalize':
        # Normalización por desviación estándar
        # Estira el rango dinámico de la imagen
        mean, std = cv2.meanStdDev(image)
        if std[0][0] == 0:
            return image
        normalized = ((image.astype(np.float32) - mean[0][0]) / (std[0][0] + 1e-8) * 50 + 128).astype(np.uint8)
        return np.clip(normalized, 0, 255).astype(np.uint8)
    
    return image
