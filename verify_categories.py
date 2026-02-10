#!/usr/bin/env python
"""Script para verificar el filtrado de categorías en dataset CCEDD."""

import os
import json

# Rutas
IMAGES_DIR = r'C:\Users\mngra\projects\AI\Pap\PAP_DATA\CCEDD\CCEDD-UTN-10imgs\train'
JSON_PATH = os.path.join(IMAGES_DIR, '_annotations.coco.json')

print("🔍 Verificando filtrado de categorías en CCEDD...\n")

# Cargar JSON
with open(JSON_PATH, 'r') as f:
    coco_data = json.load(f)

# Crear mapeo de categorías
categories_map = {cat['id']: cat['name'] for cat in coco_data['categories']}

print("📊 Categorías disponibles:")
for cat_id, cat_name in categories_map.items():
    # Contar anotaciones por categoría
    count = sum(1 for ann in coco_data['annotations'] if ann['category_id'] == cat_id)
    
    # Marcar las que se usan
    usado = "✅ USADA" if cat_id in [4, 5] else "❌ Ignorada"
    print(f"   ID {cat_id}: {cat_name:25s} - {count:4d} anotaciones - {usado}")

# Estadísticas por imagen
print("\n📷 Anotaciones por imagen:")
print(f"{'Imagen':<50s} {'Total':>8s} {'Kernels':>8s} {'Filtradas':>10s}")
print("-" * 80)

for img_info in coco_data['images']:
    img_id = img_info['id']
    fname = img_info['file_name']
    
    # Todas las anotaciones
    all_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
    
    # Solo kernels (4 y 5)
    kernel_anns = [ann for ann in all_anns if ann['category_id'] in [4, 5]]
    
    # Porcentaje
    pct = (len(kernel_anns) / len(all_anns) * 100) if all_anns else 0
    
    print(f"{fname:<50s} {len(all_anns):>8d} {len(kernel_anns):>8d} {pct:>9.1f}%")

# Totales
total_anns = len(coco_data['annotations'])
kernel_anns = sum(1 for ann in coco_data['annotations'] if ann['category_id'] in [4, 5])
pct_total = (kernel_anns / total_anns * 100) if total_anns else 0

print("-" * 80)
print(f"{'TOTAL':<50s} {total_anns:>8d} {kernel_anns:>8d} {pct_total:>9.1f}%")

print(f"\n✅ Resumen:")
print(f"   - Total de anotaciones: {total_anns}")
print(f"   - Kernels (ID 4 y 5): {kernel_anns} ({pct_total:.1f}%)")
print(f"   - Otras categorías: {total_anns - kernel_anns} ({100-pct_total:.1f}%)")
print(f"\n   El pipeline solo procesará {kernel_anns} anotaciones de núcleos.")
