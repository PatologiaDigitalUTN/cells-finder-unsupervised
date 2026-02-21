"""
Estrategias de evaluación para el pipeline de clustering.

Este módulo implementa diferentes métodos de evaluación:
- BoundingBoxStrategy: Método legacy basado en agrupamiento y overlap de bounding boxes
- SegmentationStrategy: Método basado en máscaras de segmentación con métricas DICE/IoU
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import List, Dict, Tuple, Any
from pycocotools import mask as mask_utils


class EvaluationStrategy(ABC):
    """Clase base abstracta para estrategias de evaluación."""
    
    @abstractmethod
    def load_ground_truth(self, coco_data: dict, image_id: int, category_ids: List[int], 
                         image_shape: Tuple[int, int], scale: float = 1.0) -> Any:
        """
        Carga ground truth desde COCO.
        
        Parameters:
        -----------
        coco_data : dict
            Datos COCO completos
        image_id : int
            ID de la imagen
        category_ids : list
            IDs de categorías a cargar
        image_shape : tuple
            (height, width) de la imagen (ya escalada)
        scale : float
            Factor de escala aplicado a la imagen (default: 1.0)
            
        Returns:
        --------
        Ground truth en formato específico de la estrategia
        """
        pass
    
    @abstractmethod
    def evaluate_patches(self, patches: List[dict], image: np.ndarray, 
                        ground_truth: Any) -> Dict[str, float]:
        """
        Evalúa patches contra ground truth.
        
        Parameters:
        -----------
        patches : list
            Lista de patches con posiciones
        image : np.ndarray
            Imagen
        ground_truth : Any
            Ground truth en formato de la estrategia
            
        Returns:
        --------
        dict : Métricas de evaluación
        """
        pass
    
    @abstractmethod
    def get_visualization_data(self, patches: List[dict], image: np.ndarray,
                              ground_truth: Any) -> Dict[str, Any]:
        """
        Obtiene datos para visualización.
        
        Returns:
        --------
        dict : Datos para visualizar (detecciones, GT, etc.)
        """
        pass


class BoundingBoxStrategy(EvaluationStrategy):
    """
    Estrategia legacy basada en bounding boxes.
    
    Proceso:
    1. Agrupa patches conectados
    2. Calcula envolvente rectangular
    3. Evalúa overlap con GT boxes
    """
    
    def __init__(self, cleanup_params: dict = None, grouping_params: dict = None):
        """
        Parameters:
        -----------
        cleanup_params : dict
            Parámetros para limpieza de patches
            - min_patches: Mínimo de patches por componente (default: 3)
            - dilate_px: Dilatación para conectar patches (default: 32)
        grouping_params : dict
            Parámetros para agrupamiento
            - min_patches_por_grupo: Mínimo patches por grupo (default: 3)
            - dilate_px: Dilatación para agrupar (default: 5)
        """
        self.cleanup_params = cleanup_params or {'min_patches': 3, 'dilate_px': 32}
        self.grouping_params = grouping_params or {'min_patches_por_grupo': 3, 'dilate_px': 5}
    
    def load_ground_truth(self, coco_data: dict, image_id: int, category_ids: List[int],
                         image_shape: Tuple[int, int], scale: float = 1.0) -> List[Tuple]:
        """Carga GT boxes en formato (category_id, x1, y1, x2, y2)."""
        from utils.coco_loader import load_image_and_boxes_from_coco
        
        # Cargar boxes (usando función existente)
        _, boxes, _ = load_image_and_boxes_from_coco(
            coco_path=None,  # No se usa
            images_dir=None,  # No se usa
            image_id=image_id,
            block_size=224,
            category_ids=category_ids,
            coco_data=coco_data  # Pasar datos pre-cargados
        )
        
        # Las boxes ya están escaladas por load_image_and_boxes_from_coco
        # solo retornar
        return boxes
    
    def evaluate_patches(self, patches: List[dict], image: np.ndarray,
                        ground_truth: List[Tuple]) -> Dict[str, float]:
        """Evalúa usando agrupamiento y overlap de boxes."""
        from utils.evaluation import (
            limpiar_patches_por_componentes_mask,
            agrupar_patches_en_grupos,
            evaluar_grupos_vs_boxes_plus
        )
        
        H, W = image.shape[:2]
        
        # Limpieza
        kept, removed, _ = limpiar_patches_por_componentes_mask(
            image, patches,
            min_patches=self.cleanup_params['min_patches'],
            dilate_px=self.cleanup_params['dilate_px']
        )
        
        # Agrupamiento
        grupos, _ = agrupar_patches_en_grupos(
            image, kept,
            min_patches_por_grupo=self.grouping_params['min_patches_por_grupo'],
            dilate_px=self.grouping_params['dilate_px']
        )
        
        # Evaluación
        if len(grupos) == 0:
            return {
                'groups_TP': 0,
                'groups_FP': 0,
                'group_precision': 0.0,
                'gt_recall_coverage': 0.0,
                'f1_coverage': 0.0,
                'gt_hit': 0,
                'gt_total': len(ground_truth),
                'pred_hits': [],
                'gt_covered': [],
                'grupos': [],
                'kept_patches': kept,
                'removed_patches': removed
            }
        
        metrics = evaluar_grupos_vs_boxes_plus(
            image, grupos, ground_truth, match_mode='overlap'
        )
        
        # Agregar datos adicionales para visualización
        metrics['grupos'] = grupos
        metrics['kept_patches'] = kept
        metrics['removed_patches'] = removed
        
        return metrics
    
    def get_visualization_data(self, patches: List[dict], image: np.ndarray,
                              ground_truth: List[Tuple]) -> Dict[str, Any]:
        """Obtiene grupos y boxes para visualización."""
        metrics = self.evaluate_patches(patches, image, ground_truth)
        
        return {
            'type': 'bbox',
            'groups': metrics['grupos'],
            'gt_boxes': ground_truth,
            'pred_hits': metrics['pred_hits'],
            'gt_covered': metrics['gt_covered'],
            'kept_patches': metrics['kept_patches']
        }


class SegmentationStrategy(EvaluationStrategy):
    """
    Estrategia basada en máscaras de segmentación.
    
    Proceso:
    1. Carga máscaras de polígonos COCO
    2. Crea máscara binaria de patches predichos
    3. Calcula métricas pixelwise: DICE, IoU, Precision, Recall
    """
    
    def __init__(self, merge_instances: bool = True):
        """
        Parameters:
        -----------
        merge_instances : bool
            Si True, fusiona todas las instancias en una máscara binaria
            Si False, evalúa cada instancia por separado
        """
        self.merge_instances = merge_instances
    
    def load_ground_truth(self, coco_data: dict, image_id: int, category_ids: List[int],
                         image_shape: Tuple[int, int], scale: float = 1.0) -> Dict[str, Any]:
        """Carga máscaras de segmentación desde polígonos COCO y las escala."""
        H, W = image_shape
        
        # Filtrar anotaciones
        annotations = [
            ann for ann in coco_data['annotations']
            if ann['image_id'] == image_id and ann['category_id'] in category_ids
        ]
        
        if self.merge_instances:
            # Crear máscara binaria fusionada
            merged_mask = np.zeros((H, W), dtype=np.uint8)
            
            for ann in annotations:
                # Convertir polígono a máscara
                if 'segmentation' in ann and ann['segmentation']:
                    if isinstance(ann['segmentation'], list):
                        # Formato polígono - ESCALAR COORDENADAS
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape(-1, 2)
                            # Escalar coordenadas del polígono
                            poly_scaled = (poly * scale).astype(np.int32)
                            cv2.fillPoly(merged_mask, [poly_scaled], 1)
                    elif isinstance(ann['segmentation'], dict):
                        # Formato RLE - ESCALAR MÁSCARA
                        rle = ann['segmentation']
                        mask = mask_utils.decode(rle)
                        # Redimensionar máscara
                        orig_h, orig_w = mask.shape
                        mask_scaled = cv2.resize(mask.astype(np.uint8), 
                                               (int(orig_w * scale), int(orig_h * scale)),
                                               interpolation=cv2.INTER_NEAREST)
                        # Empastar en la máscara final del tamaño correcto
                        merged_mask[:mask_scaled.shape[0], :mask_scaled.shape[1]] = \
                            np.logical_or(merged_mask[:mask_scaled.shape[0], :mask_scaled.shape[1]], 
                                        mask_scaled).astype(np.uint8)
            
            return {
                'mask': merged_mask,
                'type': 'merged',
                'n_instances': len(annotations)
            }
        else:
            # Mantener máscaras individuales
            masks = []
            for ann in annotations:
                if 'segmentation' in ann and ann['segmentation']:
                    mask = np.zeros((H, W), dtype=np.uint8)
                    
                    if isinstance(ann['segmentation'], list):
                        # Formato polígono - ESCALAR COORDENADAS
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape(-1, 2)
                            # Escalar coordenadas del polígono
                            poly_scaled = (poly * scale).astype(np.int32)
                            cv2.fillPoly(mask, [poly_scaled], 1)
                    elif isinstance(ann['segmentation'], dict):
                        # Formato RLE - ESCALAR MÁSCARA
                        rle = ann['segmentation']
                        mask_rle = mask_utils.decode(rle)
                        # Redimensionar máscara
                        orig_h, orig_w = mask_rle.shape
                        mask = cv2.resize(mask_rle.astype(np.uint8),
                                        (int(orig_w * scale), int(orig_h * scale)),
                                        interpolation=cv2.INTER_NEAREST)
                    
                    masks.append(mask)
            
            return {
                'masks': masks,
                'type': 'individual',
                'n_instances': len(masks)
            }
    
    def _create_prediction_mask(self, patches: List[dict], image_shape: Tuple[int, int]) -> np.ndarray:
        """Crea máscara binaria de patches predichos."""
        H, W = image_shape
        pred_mask = np.zeros((H, W), dtype=np.uint8)
        
        for patch in patches:
            x1, y1, x2, y2 = patch['position']
            pred_mask[y1:y2, x1:x2] = 1
        
        return pred_mask
    
    def _calculate_dice(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calcula coeficiente DICE."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        pred_sum = pred_mask.sum()
        gt_sum = gt_mask.sum()
        
        if pred_sum + gt_sum == 0:
            return 1.0  # Ambas vacías = perfecto
        
        dice = (2.0 * intersection) / (pred_sum + gt_sum)
        return float(dice)
    
    def _calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calcula Intersection over Union."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 1.0
        
        iou = intersection / union
        return float(iou)
    
    def _calculate_pixelwise_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """Calcula precisión y recall a nivel de píxel."""
        tp = np.logical_and(pred_mask, gt_mask).sum()
        fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
        fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
        tn = np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    def evaluate_patches(self, patches: List[dict], image: np.ndarray,
                        ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evalúa usando métricas de segmentación."""
        H, W = image.shape[:2]
        
        # Crear máscara de predicción
        pred_mask = self._create_prediction_mask(patches, (H, W))
        
        if ground_truth['type'] == 'merged':
            # Evaluar contra máscara fusionada
            gt_mask = ground_truth['mask']
            
            dice = self._calculate_dice(pred_mask, gt_mask)
            iou = self._calculate_iou(pred_mask, gt_mask)
            pixelwise = self._calculate_pixelwise_metrics(pred_mask, gt_mask)
            
            return {
                'dice': dice,
                'iou': iou,
                'precision': pixelwise['precision'],
                'recall': pixelwise['recall'],
                'f1_coverage': pixelwise['f1'],  # Mantener nombre consistente
                'tp_pixels': pixelwise['tp'],
                'fp_pixels': pixelwise['fp'],
                'fn_pixels': pixelwise['fn'],
                'n_instances': ground_truth['n_instances'],
                'pred_mask': pred_mask,
                'gt_mask': gt_mask
            }
        else:
            # Evaluar contra máscaras individuales (promedio)
            gt_masks = ground_truth['masks']
            
            if len(gt_masks) == 0:
                return {
                    'dice': 0.0,
                    'iou': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_coverage': 0.0,
                    'tp_pixels': 0,
                    'fp_pixels': 0,
                    'fn_pixels': 0,
                    'n_instances': 0,
                    'pred_mask': pred_mask,
                    'gt_mask': np.zeros((H, W), dtype=np.uint8)
                }
            
            # Fusionar GT para métricas globales
            gt_merged = np.zeros((H, W), dtype=np.uint8)
            for mask in gt_masks:
                gt_merged = np.logical_or(gt_merged, mask).astype(np.uint8)
            
            dice = self._calculate_dice(pred_mask, gt_merged)
            iou = self._calculate_iou(pred_mask, gt_merged)
            pixelwise = self._calculate_pixelwise_metrics(pred_mask, gt_merged)
            
            return {
                'dice': dice,
                'iou': iou,
                'precision': pixelwise['precision'],
                'recall': pixelwise['recall'],
                'f1_coverage': pixelwise['f1'],
                'tp_pixels': pixelwise['tp'],
                'fp_pixels': pixelwise['fp'],
                'fn_pixels': pixelwise['fn'],
                'n_instances': len(gt_masks),
                'pred_mask': pred_mask,
                'gt_mask': gt_merged
            }
    
    def get_visualization_data(self, patches: List[dict], image: np.ndarray,
                              ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene máscaras para visualización."""
        metrics = self.evaluate_patches(patches, image, ground_truth)
        
        # Preparar máscaras GT para visualización
        if ground_truth['type'] == 'merged':
            gt_masks = [ground_truth['mask']]
        else:
            gt_masks = ground_truth['masks']
        
        return {
            'type': 'segmentation',
            'pred_mask': metrics['pred_mask'],
            'gt_mask': metrics['gt_mask'],
            'tp_mask': np.logical_and(metrics['pred_mask'], metrics['gt_mask']).astype(np.uint8),
            'fp_mask': np.logical_and(metrics['pred_mask'], np.logical_not(metrics['gt_mask'])).astype(np.uint8),
            'fn_mask': np.logical_and(np.logical_not(metrics['pred_mask']), metrics['gt_mask']).astype(np.uint8),
            'gt_masks': gt_masks,
            'n_instances': ground_truth['n_instances']
        }


def create_evaluation_strategy(strategy_name: str, **kwargs) -> EvaluationStrategy:
    """
    Factory para crear estrategias de evaluación.
    
    Parameters:
    -----------
    strategy_name : str
        Nombre de la estrategia: 'bbox' o 'segmentation'
    **kwargs : dict
        Parámetros específicos de la estrategia
        
    Returns:
    --------
    EvaluationStrategy : Instancia de la estrategia
    """
    strategies = {
        'bbox': BoundingBoxStrategy,
        'segmentation': SegmentationStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Estrategia '{strategy_name}' no reconocida. Opciones: {list(strategies.keys())}")
    
    return strategies[strategy_name](**kwargs)
