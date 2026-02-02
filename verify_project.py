#!/usr/bin/env python
"""
Script de verificación del proyecto.
Verifica que toda la estructura esté completa y funcione.
"""

import os
import sys
from pathlib import Path

# Colores para terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def check(condition, message):
    """Print check result."""
    if condition:
        print(f"{GREEN}✅{RESET} {message}")
        return True
    else:
        print(f"{RED}❌{RESET} {message}")
        return False

def check_file(path, name=None):
    """Check if file exists."""
    name = name or path
    return check(os.path.exists(path), f"Archivo '{name}' existe")

def check_dir(path, name=None):
    """Check if directory exists."""
    name = name or path
    return check(os.path.isdir(path), f"Carpeta '{name}' existe")

def print_section(title):
    """Print section title."""
    print(f"\n{BOLD}{YELLOW}{'='*60}{RESET}")
    print(f"{BOLD}{YELLOW}{title}{RESET}")
    print(f"{BOLD}{YELLOW}{'='*60}{RESET}")

def main():
    """Run verification."""
    base_dir = Path(__file__).parent
    os.chdir(base_dir)
    
    print(f"\n{BOLD}{GREEN}VERIFICACIÓN DE PROYECTO{RESET}")
    print(f"Ubicación: {base_dir}\n")
    
    # ========== Estructura de carpetas ==========
    print_section("1. ESTRUCTURA DE CARPETAS")
    
    results = []
    results.append(check_dir("utils", "Carpeta 'utils'"))
    results.append(check_dir("utils", "Módulos en 'utils'"))
    
    # ========== Archivos principales ==========
    print_section("2. ARCHIVOS PRINCIPALES")
    
    results.append(check_file("main.py", "main.py (script principal)"))
    results.append(check_file("config.py", "config.py (configuración)"))
    results.append(check_file("example.py", "example.py (tutorial)"))
    results.append(check_file("requirements.txt", "requirements.txt (dependencias)"))
    results.append(check_file("README.md", "README.md (documentación)"))
    results.append(check_file("REFACTOR_NOTES.md", "REFACTOR_NOTES.md (notas de refactorización)"))
    
    # ========== Módulos utils ==========
    print_section("3. MÓDULOS UTILITIES")
    
    results.append(check_file("utils/__init__.py", "utils/__init__.py"))
    results.append(check_file("utils/image_processing.py", "utils/image_processing.py"))
    results.append(check_file("utils/embeddings.py", "utils/embeddings.py"))
    results.append(check_file("utils/multi_step_clustering.py", "utils/multi_step_clustering.py"))
    results.append(check_file("utils/evaluation.py", "utils/evaluation.py"))
    results.append(check_file("utils/visualization.py", "utils/visualization.py"))
    
    # ========== Imports ==========
    print_section("4. VERIFICACIÓN DE IMPORTS")
    
    try:
        import config
        results.append(check(True, "Importar 'config' módulo"))
    except Exception as e:
        results.append(check(False, f"Importar 'config' módulo: {e}"))
    
    try:
        from utils.image_processing import load_image_and_boxes_from_json_cropped, BETHESDA_CLASSES
        results.append(check(True, "Importar 'utils.image_processing'"))
    except Exception as e:
        results.append(check(False, f"Importar 'utils.image_processing': {e}"))
    
    try:
        from utils.embeddings import get_all_patch_embeddings_from_image
        results.append(check(True, "Importar 'utils.embeddings'"))
    except Exception as e:
        results.append(check(False, f"Importar 'utils.embeddings': {e}"))
    
    try:
        from utils.multi_step_clustering import run_block_clustering_on_embeddings
        results.append(check(True, "Importar 'utils.multi_step_clustering'"))
    except Exception as e:
        results.append(check(False, f"Importar 'utils.multi_step_clustering': {e}"))
    
    try:
        from utils.evaluation import evaluar_grupos_vs_boxes_plus
        results.append(check(True, "Importar 'utils.evaluation'"))
    except Exception as e:
        results.append(check(False, f"Importar 'utils.evaluation': {e}"))
    
    try:
        from utils.visualization import visualizar_clusters_basicos
        results.append(check(True, "Importar 'utils.visualization'"))
    except Exception as e:
        results.append(check(False, f"Importar 'utils.visualization': {e}"))
    
    # ========== Respaldo ==========
    print_section("5. ARCHIVO DE RESPALDO")
    
    results.append(check_file(
        "clustering_segmentation_biomedclip.py.bak",
        "Archivo original (backup)"
    ))
    
    # ========== Resumen ==========
    print_section("RESUMEN")
    
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    print(f"\n{BOLD}Verificaciones: {passed}/{total}{RESET}")
    
    if failed == 0:
        print(f"{GREEN}{BOLD}✅ TODAS LAS VERIFICACIONES PASARON{RESET}")
        print(f"\n{YELLOW}Próximos pasos:{RESET}")
        print(f"  1. Editar config.py con tus rutas de datos")
        print(f"  2. Ejecutar: python main.py")
        print(f"  3. O ver tutorial: python example.py")
        return 0
    else:
        print(f"{RED}{BOLD}❌ {failed} VERIFICACIONES FALLARON{RESET}")
        print(f"\nRevisar errores arriba.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
