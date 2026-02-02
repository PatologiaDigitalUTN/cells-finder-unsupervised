"""
Utilidades para extraer embeddings y características usando BiomedCLIP.
"""

import torch
import numpy as np
from PIL import Image


def get_patch_embeddings_grid(tile_np, model, preprocess, normalize=True):
    """
    Extrae una grilla 2D de embeddings de patches desde un tile.
    Devuelve matriz [grid_y, grid_x, embedding_dim].
    
    Parameters:
    -----------
    tile_np : np.ndarray
        Tile de imagen.
    model : torch.nn
        Modelo BiomedCLIP.
    preprocess : callable
        Función de preprocessing del modelo.
    normalize : bool
        Si normalizar embeddings a L2=1.
    
    Returns:
    --------
    grid : np.ndarray
        Grilla de embeddings, shape (grid_size, grid_size, D).
        None si hay error.
    """
    try:
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

        return patch_tokens.reshape(grid_size, grid_size, -1)

    except Exception as e:
        print(f"Error en get_patch_embeddings_grid: {e}")
        return None


def get_all_patch_embeddings_from_image(img, model, preprocess, tile_size=224, stride=224):
    """
    Extrae embeddings de todos los patches de una imagen completa.
    
    Parameters:
    -----------
    img : np.ndarray
        Imagen de entrada.
    model : torch.nn
        Modelo BiomedCLIP.
    preprocess : callable
        Función de preprocessing.
    tile_size : int
        Tamaño de cada tile.
    stride : int
        Stride entre tiles (puede tener solapamiento si stride < tile_size).
    
    Returns:
    --------
    patch_data : list[dict]
        Lista de dicts con 'embedding' y 'position' para cada patch.
    """
    h, w = img.shape[:2]
    patch_data = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = img[y:y+tile_size, x:x+tile_size]
            grid = get_patch_embeddings_grid(tile, model, preprocess)
            
            if grid is None:
                continue

            gs_y, gs_x, D = grid.shape
            for i in range(gs_y):
                for j in range(gs_x):
                    emb = grid[i, j]
                    patch_x = x + j * (tile_size // gs_x)
                    patch_y = y + i * (tile_size // gs_y)
                    patch_data.append({
                        'embedding': emb,
                        'position': (
                            patch_x, 
                            patch_y, 
                            patch_x + (tile_size // gs_x), 
                            patch_y + (tile_size // gs_y)
                        )
                    })

    return patch_data
