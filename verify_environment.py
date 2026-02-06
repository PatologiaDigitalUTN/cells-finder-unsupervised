#!/usr/bin/env python
"""
Script de verificación del ambiente local.
Verifica que todas las dependencias estén disponibles.
"""

import sys
import importlib

def check_import(module_name, display_name=None):
    """Intenta importar un módulo y reporta el estado."""
    display_name = display_name or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'desconocida')
        print(f"[OK] {display_name:30s} - v{version}")
        return True
    except ImportError as e:
        print(f"[FAIL] {display_name:30s} - ERROR: {str(e)}")
        return False

def main():
    print("=" * 70)
    print("VERIFICACIÓN DE AMBIENTE LOCAL")
    print("=" * 70)
    
    print(f"\n[INFO] Python {sys.version}")
    print(f"[PATH] Ejecutable: {sys.executable}\n")
    
    print("Verificando dependencias críticas:")
    print("-" * 70)
    
    critical = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-Learn'),
        ('matplotlib', 'Matplotlib'),
        ('PIL', 'Pillow'),
    ]
    
    critical_ok = all(check_import(m, d) for m, d in critical)
    
    print("\nVerificando dependencias específicas del proyecto:")
    print("-" * 70)
    
    project = [
        ('open_clip', 'OpenCLIP'),
        ('transformers', 'Transformers'),
        ('kagglehub', 'KaggleHub'),
    ]
    
    project_ok = all(check_import(m, d) for m, d in project)
    
    print("\nVerificando Jupyter:")
    print("-" * 70)
    
    jupyter_ok = check_import('jupyter', 'Jupyter') and check_import('ipykernel', 'IPyKernel')
    
    print("\nVerificando módulos locales:")
    print("-" * 70)
    
    local = [
        ('utils', 'utils (módulo local)'),
        ('utils.embeddings', 'utils.embeddings'),
        ('utils.image_processing', 'utils.image_processing'),
        ('utils.multi_step_clustering', 'utils.multi_step_clustering'),
        ('utils.evaluation', 'utils.evaluation'),
        ('utils.visualization', 'utils.visualization'),
    ]
    
    local_ok = all(check_import(m, d) for m, d in local)
    
    print("\n" + "=" * 70)
    if critical_ok and project_ok and jupyter_ok and local_ok:
        print("[OK] AMBIENTE LISTO - Todas las dependencias están disponibles")
        print("=" * 70)
        print("\n[NEXT] Próximo paso: ejecuta 'jupyter notebook kaggle_notebook.ipynb'")
        return 0
    else:
        print("[WARN] FALTA INSTALAR ALGUNAS DEPENDENCIAS")
        print("=" * 70)
        print("\nIntenta ejecutar:")
        print("  pip install -r requirements.txt")
        print("  pip install jupyter ipykernel")
        return 1

if __name__ == '__main__':
    sys.exit(main())
