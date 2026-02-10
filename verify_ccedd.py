#!/usr/bin/env python
"""Script para verificar dataset CCEDD en formato COCO."""

import os
import json

# Verificar rutas
IMAGES_DIR = r'C:\Users\mngra\projects\AI\Pap\PAP_DATA\CCEDD\CCEDD-UTN-10imgs\train'
JSON_PATH = os.path.join(IMAGES_DIR, '_annotations.coco.json')

print("🔍 Verificando dataset CCEDD...\n")

# Verificar carpeta
if not os.path.exists(IMAGES_DIR):
    print(f"❌ ERROR: No existe la carpeta: {IMAGES_DIR}")
    print("   Por favor verificar la ruta.")
    exit(1)
else:
    print(f"✅ Carpeta existe: {IMAGES_DIR}")

# Listar archivos en la carpeta
files = os.listdir(IMAGES_DIR)
print(f"\n📂 Archivos en la carpeta: {len(files)} archivos")
images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
json_files = [f for f in files if f.lower().endswith('.json')]

print(f"   - Imágenes: {len(images)}")
if images:
    for img in images[:5]:
        print(f"     • {img}")
    if len(images) > 5:
        print(f"     ... y {len(images)-5} más")

print(f"   - JSON files: {len(json_files)}")
for jf in json_files:
    print(f"     • {jf}")

# Verificar JSON de anotaciones
if not os.path.exists(JSON_PATH):
    print(f"\n❌ No encontrado: {JSON_PATH}")
    if json_files:
        print(f"   Posibles opciones:")
        for jf in json_files:
            print(f"     • {jf}")
            alt_path = os.path.join(IMAGES_DIR, jf)
            with open(alt_path, 'r') as f:
                try:
                    data = json.load(f)
                    if 'images' in data and 'annotations' in data:
                        print(f"       → Este parece ser COCO format ✅")
                    else:
                        print(f"       → No es COCO format (keys: {list(data.keys())})")
                except Exception as e:
                    print(f"       → Error al leer: {e}")
    else:
        print("   No hay archivos JSON en la carpeta.")
        print("   ⚠️ Necesitas el archivo de anotaciones COCO.")
    exit(1)

# Cargar y verificar formato COCO
print(f"\n✅ Archivo de anotaciones encontrado: {JSON_PATH}")
with open(JSON_PATH, 'r') as f:
    coco_data = json.load(f)

print("\n📊 Estructura del JSON:")
print(f"   - images: {len(coco_data.get('images', []))}")
print(f"   - annotations: {len(coco_data.get('annotations', []))}")
print(f"   - categories: {len(coco_data.get('categories', []))}")

if 'categories' in coco_data:
    print("\n🏷️  Categorías:")
    for cat in coco_data['categories']:
        print(f"   ID {cat['id']}: {cat['name']}")

if 'images' in coco_data and len(coco_data['images']) > 0:
    print("\n🖼️  Ejemplos de imágenes:")
    for img_info in coco_data['images'][:3]:
        img_id = img_info['id']
        fname = img_info['file_name']
        w, h = img_info.get('width', '?'), img_info.get('height', '?')
        
        # Contar anotaciones
        anns = [a for a in coco_data['annotations'] if a['image_id'] == img_id]
        
        print(f"   ID {img_id}: {fname} ({w}x{h}) - {len(anns)} anotaciones")
        
        # Verificar tipo de anotaciones
        if anns:
            ann = anns[0]
            has_seg = 'segmentation' in ann and ann['segmentation']
            has_bbox = 'bbox' in ann and ann['bbox']
            print(f"          Segmentation: {'✅' if has_seg else '❌'}")
            print(f"          BBox: {'✅' if has_bbox else '❌'}")

print("\n✅ Dataset CCEDD verificado correctamente!")
print("   Puedes ejecutar la notebook kaggle_notebook.ipynb")
