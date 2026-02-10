"""
Utilidades para cargar datos en formato COCO (Common Objects in Context).
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path


def load_coco_annotations(json_path):
    """
    Carga anotaciones desde un archivo JSON en formato COCO.
    
    Parameters:
    -----------
    json_path : str
        Ruta al archivo JSON de COCO.
    
    Returns:
    --------
    data : dict
        Diccionario con las anotaciones COCO completas.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def segmentation_to_bbox(segmentation):
    """
    Convierte una segmentación (polígono) a bounding box.
    
    Parameters:
    -----------
    segmentation : list
        Lista de coordenadas [x1, y1, x2, y2, ...] del polígono.
    
    Returns:
    --------
    bbox : tuple
        (x_min, y_min, x_max, y_max)
    """
    if isinstance(segmentation, list) and len(segmentation) > 0:
        # Si es una lista de polígonos, tomar el primero
        if isinstance(segmentation[0], list):
            coords = segmentation[0]
        else:
            coords = segmentation
        
        # Extraer coordenadas x e y
        xs = coords[0::2]
        ys = coords[1::2]
        
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        
        return (x_min, y_min, x_max, y_max)
    
    return None


def coco_bbox_to_xyxy(bbox):
    """
    Convierte bbox de formato COCO [x, y, width, height] a [x1, y1, x2, y2].
    
    Parameters:
    -----------
    bbox : list or tuple
        [x, y, width, height] en formato COCO.
    
    Returns:
    --------
    xyxy : tuple
        (x1, y1, x2, y2)
    """
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def load_image_and_boxes_from_coco(
    json_path,
    images_dir,
    image_id=None,
    image_filename=None,
    boxes_size=60,
    block_size=224,
    category_ids=None
):
    """
    Carga imagen y bounding boxes desde anotaciones COCO.
    
    Parameters:
    -----------
    json_path : str
        Ruta al archivo JSON de COCO.
    images_dir : str
        Directorio donde están las imágenes.
    image_id : int, optional
        ID de la imagen en COCO. Si no se provee, se usa image_filename o la primera imagen.
    image_filename : str, optional
        Nombre del archivo de imagen. Alternativa a image_id.
    boxes_size : int
        Tamaño de box alrededor del centro (para compatibilidad, no se usa con COCO).
    block_size : int
        Tamaño de bloque para recorte (imagen será múltiplo de esto).
    category_ids : list, optional
        Lista de IDs de categorías a filtrar. Si None, toma todas.
    
    Returns:
    --------
    img_crop : np.ndarray
        Imagen recortada a múltiplos de block_size.
    boxes : list
        Lista de boxes en formato (category_id, x1, y1, x2, y2).
    image_filename : str
        Nombre del archivo de imagen.
    """
    # Cargar JSON
    coco_data = load_coco_annotations(json_path)
    
    # Construir mapeos
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Seleccionar imagen
    if image_id is not None:
        if image_id not in images_dict:
            raise ValueError(f"Image ID {image_id} no encontrado en COCO JSON")
        image_info = images_dict[image_id]
    elif image_filename is not None:
        # Buscar por nombre de archivo
        image_info = None
        for img in coco_data['images']:
            if img['file_name'] == image_filename:
                image_info = img
                break
        if image_info is None:
            raise ValueError(f"Imagen '{image_filename}' no encontrada en COCO JSON")
    else:
        # Tomar la primera imagen
        image_info = coco_data['images'][0]
    
    selected_image_id = image_info['id']
    image_file = image_info['file_name']
    
    # Cargar imagen
    img_path = os.path.join(images_dir, image_file)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Imagen no encontrada: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    
    # Recortar a múltiplos de block_size
    Hc = (H // block_size) * block_size
    Wc = (W // block_size) * block_size
    
    if Hc == 0 or Wc == 0:
        raise ValueError(f"Imagen {W}x{H} es muy chica para block_size={block_size}.")
    
    img_crop = img_rgb[:Hc, :Wc]
    
    # Obtener anotaciones para esta imagen
    annotations = [ann for ann in coco_data['annotations'] 
                   if ann['image_id'] == selected_image_id]
    
    # Filtrar por categorías si se especifica
    if category_ids is not None:
        annotations = [ann for ann in annotations if ann['category_id'] in category_ids]
    
    # Construir boxes
    boxes = []
    for ann in annotations:
        category_id = ann['category_id']
        
        # Priorizar bbox si existe
        if 'bbox' in ann and ann['bbox']:
            x1, y1, x2, y2 = coco_bbox_to_xyxy(ann['bbox'])
        elif 'segmentation' in ann and ann['segmentation']:
            # Convertir segmentación a bbox
            bbox_from_seg = segmentation_to_bbox(ann['segmentation'])
            if bbox_from_seg is None:
                continue
            x1, y1, x2, y2 = bbox_from_seg
        else:
            continue
        
        # Ajustar a la imagen recortada
        x1 = max(0, min(x1, Wc))
        x2 = max(0, min(x2, Wc))
        y1 = max(0, min(y1, Hc))
        y2 = max(0, min(y2, Hc))
        
        # Verificar que el box es válido
        if x2 > x1 and y2 > y1:
            boxes.append((category_id, int(x1), int(y1), int(x2), int(y2)))
    
    return img_crop, boxes, image_file


def list_coco_images(json_path):
    """
    Lista todas las imágenes disponibles en el archivo COCO.
    
    Parameters:
    -----------
    json_path : str
        Ruta al archivo JSON de COCO.
    
    Returns:
    --------
    images_list : list of dict
        Lista con información de cada imagen: {'id', 'file_name', 'width', 'height'}
    """
    coco_data = load_coco_annotations(json_path)
    return coco_data['images']


def get_coco_categories(json_path):
    """
    Obtiene las categorías definidas en el archivo COCO.
    
    Parameters:
    -----------
    json_path : str
        Ruta al archivo JSON de COCO.
    
    Returns:
    --------
    categories : dict
        Diccionario {category_id: category_name}
    """
    coco_data = load_coco_annotations(json_path)
    return {cat['id']: cat['name'] for cat in coco_data['categories']}
