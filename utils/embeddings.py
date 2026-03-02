"""
Utilidades para extraer embeddings y características usando modelos fundacionales.

Soporta múltiples modelos a través de la abstracción en model_factory.py
"""

import torch
import numpy as np
from PIL import Image


def get_patch_embeddings_grid(tile_np, model, preprocess=None, normalize=True):
    """
    Extrae una grilla 2D de embeddings de patches desde un tile.
    Devuelve matriz [grid_y, grid_x, embedding_dim].
    
    Parameters:
    -----------
    tile_np : np.ndarray
        Tile de imagen.
    model : BaseEmbeddingModel o torch.nn.Module
        Modelo de embeddings. Puede ser una instancia de BaseEmbeddingModel
        (nueva abstracción) o un modelo legacy (BiomedCLIP directo).
    preprocess : callable, optional
        Función de preprocessing del modelo (solo para legacy).
    normalize : bool
        Si normalizar embeddings a L2=1.
    
    Returns:
    --------
    grid : np.ndarray
        Grilla de embeddings, shape (grid_size, grid_size, D).
        None si hay error.
    """
    try:
        # Nueva API: usar abstracción de model_factory
        from .model_factory import BaseEmbeddingModel
        if isinstance(model, BaseEmbeddingModel):
            return model.extract_patch_embeddings(tile_np, normalize=normalize)
        
        # Legacy API: BiomedCLIP directo (backwards compatibility)
        if preprocess is None:
            raise ValueError("preprocess es requerido para modelos legacy")
        
        image = Image.fromarray(tile_np)
        image_tensor = preprocess(image.convert('RGB')).unsqueeze(0).to(
            next(model.parameters()).device
        )

        with torch.no_grad():
            tokens = model.visual.trunk.forward_features(image_tensor)  # [1, 197, D]

        if tokens.ndim != 3 or tokens.shape[1] < 2:
            return None

        patch_tokens = tokens[0, 1:]  # Sin CLS token: [196, D]

        if normalize:
            patch_tokens = patch_tokens / patch_tokens.norm(dim=1, keepdim=True)

        n_patches = patch_tokens.shape[0]
        grid_size = int(n_patches**0.5)

        if grid_size * grid_size != n_patches:
            return None

        return patch_tokens.cpu().numpy().reshape(grid_size, grid_size, -1)

    except Exception as e:
        print(f"Error en get_patch_embeddings_grid: {e}")
        return None


def get_all_patch_embeddings_from_image(
    img,
    model,
    preprocess=None,
    tile_size=224,
    stride=None,
    margin_tokens=3,
    pad_mode="reflect",
    pad_tokens=None,
):
    """
    Extrae embeddings de todos los patches de una imagen completa.
    
    Parameters:
    -----------
    img : np.ndarray
        Imagen de entrada.
    model : BaseEmbeddingModel o torch.nn.Module
        Modelo de embeddings. Puede ser una instancia de BaseEmbeddingModel
        (nueva abstracción) o un modelo legacy (BiomedCLIP directo).
    preprocess : callable, optional
        Función de preprocessing (solo para legacy).
    tile_size : int, optional
        Tamaño de cada tile. Si None y model es BaseEmbeddingModel,
        usa model.tile_size.
    stride : int, optional
        Stride entre tiles (puede tener solapamiento si stride < tile_size).
        Si es None, se ajusta automaticamente para cubrir solo la zona valida.
    margin_tokens : int
        Margen (en tokens) a descartar por borde. Usa solo el centro del grid.
    pad_mode : str or None
        Modo de padding para recuperar bordes. Usa None para desactivar.
    pad_tokens : int or None
        Padding en tokens por lado. Si es None, usa margin_tokens.
    
    Returns:
    --------
    patch_data : list[dict]
        Lista de dicts con 'embedding' y 'position' para cada patch.
    """
    # Auto-detectar tile_size si es posible
    from .model_factory import BaseEmbeddingModel
    if isinstance(model, BaseEmbeddingModel) and tile_size is None:
        tile_size = model.tile_size
    
    orig_h, orig_w = img.shape[:2]
    patch_data = []

    if margin_tokens < 0:
        raise ValueError("margin_tokens debe ser >= 0")

    patch_px0 = None
    if stride is None or (pad_mode is not None and (pad_tokens is None or pad_tokens > 0)):
        if orig_h < tile_size or orig_w < tile_size:
            return patch_data

        tile0 = img[0:tile_size, 0:tile_size]
        grid0 = get_patch_embeddings_grid(tile0, model, preprocess=preprocess)
        if grid0 is None:
            return patch_data

        gs_y0, gs_x0, _ = grid0.shape
        if margin_tokens * 2 >= gs_y0 or margin_tokens * 2 >= gs_x0:
            return patch_data

        patch_px0 = tile_size // gs_x0

    if stride is None:
        valid_px = tile_size - 2 * margin_tokens * patch_px0
        stride = max(1, valid_px)

    if pad_mode is not None:
        if pad_tokens is None:
            pad_tokens = margin_tokens

        if pad_tokens > 0:
            pad_px = pad_tokens * patch_px0
            safe_mode = pad_mode
            if pad_mode in ("reflect", "symmetric") and pad_px >= min(orig_h, orig_w):
                safe_mode = "edge"

            if img.ndim == 2:
                img = np.pad(img, ((pad_px, pad_px), (pad_px, pad_px)), mode=safe_mode)
            else:
                img = np.pad(
                    img,
                    ((pad_px, pad_px), (pad_px, pad_px), (0, 0)),
                    mode=safe_mode,
                )

    h, w = img.shape[:2]

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = img[y:y+tile_size, x:x+tile_size]
            grid = get_patch_embeddings_grid(tile, model, preprocess=preprocess)
            
            if grid is None:
                continue

            gs_y, gs_x, D = grid.shape
            if margin_tokens * 2 >= gs_y or margin_tokens * 2 >= gs_x:
                continue

            patch_px = tile_size // gs_x
            y_start = margin_tokens
            y_end = gs_y - margin_tokens
            x_start = margin_tokens
            x_end = gs_x - margin_tokens

            for i in range(y_start, y_end):
                for j in range(x_start, x_end):
                    emb = grid[i, j]
                    patch_x = x + j * patch_px
                    patch_y = y + i * patch_px
                    if pad_mode is not None and pad_tokens and pad_tokens > 0:
                        patch_x -= pad_tokens * patch_px0
                        patch_y -= pad_tokens * patch_px0

                    if (
                        patch_x < 0
                        or patch_y < 0
                        or patch_x + patch_px > orig_w
                        or patch_y + patch_px > orig_h
                    ):
                        continue
                    patch_data.append({
                        'embedding': emb,
                        'position': (
                            patch_x, 
                            patch_y, 
                            patch_x + patch_px, 
                            patch_y + patch_px
                        )
                    })

    return patch_data