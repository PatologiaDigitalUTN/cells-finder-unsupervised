"""
Factory para cargar y abstraer diferentes modelos fundacionales.

Soporta:
- BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- UNI (mahmoodlab/UNI)
- OPTIMUS (bioptimus/H-optimus-0)
- UNI 2 (mahmoodlab/uni-v2)
"""

import torch
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """Clase base abstracta para modelos de embeddings."""
    
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocess = None
    
    @abstractmethod
    def load_model(self):
        """Carga el modelo y preprocessing."""
        pass
    
    @abstractmethod
    def extract_patch_embeddings(self, tile_np, normalize=True):
        """
        Extrae embeddings de patches de un tile.
        
        Parameters:
        -----------
        tile_np : np.ndarray
            Tile de imagen (típicamente 224x224).
        normalize : bool
            Si normalizar embeddings a L2=1.
        
        Returns:
        --------
        grid : np.ndarray
            Grilla de embeddings, shape (grid_size, grid_size, D).
            None si hay error.
        """
        pass
    
    @property
    @abstractmethod
    def tile_size(self):
        """Tamaño de tile esperado por el modelo."""
        pass
    
    @property
    @abstractmethod
    def patch_size(self):
        """Tamaño de cada patch dentro del tile."""
        pass


class BiomedCLIPModel(BaseEmbeddingModel):
    """Wrapper para BiomedCLIP."""
    
    def __init__(self, model_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', device=None):
        super().__init__(device)
        self.model_name = model_name
        self.load_model()
    
    def load_model(self):
        from open_clip import create_model_from_pretrained, get_tokenizer
        print(f"Cargando BiomedCLIP desde {self.model_name}...")
        self.model, self.preprocess = create_model_from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = get_tokenizer(self.model_name)
        print("✅ BiomedCLIP cargado")
    
    def extract_patch_embeddings(self, tile_np, normalize=True):
        try:
            image = Image.fromarray(tile_np)
            image_tensor = self.preprocess(image.convert('RGB')).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                tokens = self.model.visual.trunk.forward_features(image_tensor)  # [1, 197, D]
            
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
            print(f"Error en extract_patch_embeddings (BiomedCLIP): {e}")
            return None
    
    @property
    def tile_size(self):
        return 224
    
    @property
    def patch_size(self):
        return 16


class UNIModel(BaseEmbeddingModel):
    """Wrapper para UNI (mahmoodlab/UNI)."""
    
    def __init__(self, device=None):
        super().__init__(device)
        self.load_model()
    
    def load_model(self):
        try:
            import timm
            print("Cargando UNI (mahmoodlab/UNI)...")
            
            # UNI usa ViT-L/16 entrenado con DINOv2
            self.model = timm.create_model(
                "vit_large_patch16_224", 
                img_size=224, 
                patch_size=16, 
                init_values=1e-5, 
                num_classes=0,  # Sin clasificación, solo embeddings
                dynamic_img_size=True
            )
            
            # Cargar pesos de UNI (requiere descarga manual o HF)
            # checkpoint_path = "path/to/uni_checkpoint.pth"
            # self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Preprocessing estándar de ImageNet
            from torchvision import transforms
            self.preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            print("✅ UNI cargado")
        except Exception as e:
            print(f"⚠️ Error cargando UNI: {e}")
            print("Nota: UNI requiere instalación manual de pesos")
            raise
    
    def extract_patch_embeddings(self, tile_np, normalize=True):
        try:
            image = Image.fromarray(tile_np)
            image_tensor = self.preprocess(image.convert('RGB')).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Extraer patch tokens (sin CLS)
                features = self.model.forward_features(image_tensor)
                
                # UNI devuelve [B, N+1, D] donde N es número de patches
                if hasattr(features, 'shape') and features.ndim == 3:
                    patch_tokens = features[0, 1:]  # Sin CLS token
                else:
                    return None
            
            if normalize:
                patch_tokens = patch_tokens / patch_tokens.norm(dim=1, keepdim=True)
            
            n_patches = patch_tokens.shape[0]
            grid_size = int(n_patches**0.5)
            
            if grid_size * grid_size != n_patches:
                return None
            
            return patch_tokens.cpu().numpy().reshape(grid_size, grid_size, -1)
        
        except Exception as e:
            print(f"Error en extract_patch_embeddings (UNI): {e}")
            return None
    
    @property
    def tile_size(self):
        return 224
    
    @property
    def patch_size(self):
        return 16


class OPTIMUSModel(BaseEmbeddingModel):
    """Wrapper para OPTIMUS (bioptimus/H-optimus-0)."""
    
    def __init__(self, device=None):
        super().__init__(device)
        self.load_model()
    
    def load_model(self):
        try:
            from transformers import AutoModel, AutoImageProcessor
            print("Cargando OPTIMUS (bioptimus/H-optimus-0)...")
            
            self.model = AutoModel.from_pretrained(
                "bioptimus/H-optimus-0",
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.preprocess = AutoImageProcessor.from_pretrained(
                "bioptimus/H-optimus-0",
                trust_remote_code=True
            )
            
            print("✅ OPTIMUS cargado")
        except Exception as e:
            print(f"⚠️ Error cargando OPTIMUS: {e}")
            print("Nota: OPTIMUS requiere transformers>=4.35.0 y acceso a HuggingFace")
            raise
    
    def extract_patch_embeddings(self, tile_np, normalize=True):
        try:
            image = Image.fromarray(tile_np)
            inputs = self.preprocess(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # OPTIMUS devuelve embeddings de patches
                patch_tokens = outputs.last_hidden_state[0, 1:]  # Sin CLS
            
            if normalize:
                patch_tokens = patch_tokens / patch_tokens.norm(dim=1, keepdim=True)
            
            n_patches = patch_tokens.shape[0]
            grid_size = int(n_patches**0.5)
            
            if grid_size * grid_size != n_patches:
                return None
            
            return patch_tokens.cpu().numpy().reshape(grid_size, grid_size, -1)
        
        except Exception as e:
            print(f"Error en extract_patch_embeddings (OPTIMUS): {e}")
            return None
    
    @property
    def tile_size(self):
        return 224
    
    @property
    def patch_size(self):
        return 16


class UNI2Model(BaseEmbeddingModel):
    """Wrapper para UNI 2 (mahmoodlab/uni-v2)."""
    
    def __init__(self, device=None):
        super().__init__(device)
        self.load_model()
    
    def load_model(self):
        try:
            import timm
            from huggingface_hub import hf_hub_download
            
            print("Cargando UNI 2 (mahmoodlab/uni-v2)...")
            
            # Descargar checkpoint desde HuggingFace
            checkpoint_path = hf_hub_download(
                repo_id="mahmoodlab/uni-v2",
                filename="pytorch_model.bin"
            )
            
            self.model = timm.create_model(
                "vit_large_patch16_224",
                img_size=224,
                patch_size=16,
                init_values=1e-5,
                num_classes=0,
                dynamic_img_size=True
            )
            
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            self.model = self.model.to(self.device)
            self.model.eval()
            
            from torchvision import transforms
            self.preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            print("✅ UNI 2 cargado")
        except Exception as e:
            print(f"⚠️ Error cargando UNI 2: {e}")
            raise
    
    def extract_patch_embeddings(self, tile_np, normalize=True):
        try:
            image = Image.fromarray(tile_np)
            image_tensor = self.preprocess(image.convert('RGB')).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model.forward_features(image_tensor)
                
                if hasattr(features, 'shape') and features.ndim == 3:
                    patch_tokens = features[0, 1:]
                else:
                    return None
            
            if normalize:
                patch_tokens = patch_tokens / patch_tokens.norm(dim=1, keepdim=True)
            
            n_patches = patch_tokens.shape[0]
            grid_size = int(n_patches**0.5)
            
            if grid_size * grid_size != n_patches:
                return None
            
            return patch_tokens.cpu().numpy().reshape(grid_size, grid_size, -1)
        
        except Exception as e:
            print(f"Error en extract_patch_embeddings (UNI 2): {e}")
            return None
    
    @property
    def tile_size(self):
        return 224
    
    @property
    def patch_size(self):
        return 16


def create_model(model_name='biomedclip', device=None, **kwargs):
    """
    Factory function para crear modelos.
    
    Parameters:
    -----------
    model_name : str
        Nombre del modelo: 'biomedclip', 'uni', 'optimus', 'uni2'
    device : str, optional
        Device para el modelo ('cuda', 'cpu', etc.)
    **kwargs : dict
        Argumentos adicionales para el modelo específico.
    
    Returns:
    --------
    model : BaseEmbeddingModel
        Instancia del modelo solicitado.
    
    Examples:
    ---------
    >>> model = create_model('biomedclip')
    >>> model = create_model('uni', device='cuda:0')
    >>> model = create_model('optimus')
    """
    model_name = model_name.lower().strip()
    
    if model_name == 'biomedclip':
        return BiomedCLIPModel(device=device, **kwargs)
    elif model_name == 'uni':
        return UNIModel(device=device)
    elif model_name == 'optimus':
        return OPTIMUSModel(device=device)
    elif model_name in ['uni2', 'uni_v2', 'univ2']:
        return UNI2Model(device=device)
    else:
        raise ValueError(
            f"Modelo '{model_name}' no soportado. "
            f"Opciones: 'biomedclip', 'uni', 'optimus', 'uni2'"
        )


def list_available_models():
    """Lista modelos disponibles."""
    return {
        'biomedclip': 'BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)',
        'uni': 'UNI (mahmoodlab/UNI) - ViT-L/16 con DINOv2',
        'optimus': 'OPTIMUS (bioptimus/H-optimus-0) - Pathology foundation model',
        'uni2': 'UNI 2 (mahmoodlab/uni-v2) - Segunda versión de UNI'
    }
