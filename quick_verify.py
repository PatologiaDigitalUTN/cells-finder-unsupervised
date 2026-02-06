#!/usr/bin/env python
"""Verificación rápida del ambiente."""

import sys

print("=" * 70)
print("Verificacion del Ambiente Local")
print("=" * 70)
print(f"\nPython: {sys.version}")
print(f"Ejecutable: {sys.executable}\n")

# Verificación rápida de imports críticos
modules = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'Scikit-Learn',
    'matplotlib': 'Matplotlib',
    'PIL': 'Pillow',
    'open_clip': 'OpenCLIP',
    'transformers': 'Transformers',
    'kagglehub': 'KaggleHub',
    'jupyter': 'Jupyter',
    'ipykernel': 'IPyKernel',
}

failed = []
for mod_name, display in modules.items():
    try:
        __import__(mod_name)
        print(f"[OK]   {display}")
    except ImportError as e:
        print(f"[FAIL] {display} - {str(e)[:50]}")
        failed.append(display)

print("\n" + "=" * 70)
if not failed:
    print("LISTO: Todas las dependencias estan disponibles")
    print("\nProximo paso: jupyter notebook kaggle_notebook.ipynb")
else:
    print(f"FALTA INSTALAR: {', '.join(failed)}")
    print("\nIntenta: pip install -r requirements.txt")
print("=" * 70)
