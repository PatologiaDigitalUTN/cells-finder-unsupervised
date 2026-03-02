import json

with open('kaggle_notebook_uni_hybrid.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("\n" + "="*80)
print("VISUALIZATION CELLS VERIFICATION")
print("="*80 + "\n")

vis_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'VIS' in source or 'PASO 8' in source:
            has_img_color = 'img_color = cv2.cvtColor(img,' in source
            first_line = source.split('\n')[0][:55]
            vis_cells.append((i, first_line, has_img_color))
            status = "✓ FIXED" if has_img_color else "✗ NEEDS FIX"
            print(f"{status} | Cell {i:2d} | {first_line}")

print("\n" + "="*80)
if all(status for _, _, status in vis_cells):
    print(f"✅ SUCCESS: All {len(vis_cells)} visualization cells have img_color definitions!")
else:
    broken = [i for i, _, status in vis_cells if not status]
    print(f"❌ ISSUE: Cells {broken} still need img_color definitions")
print("="*80 + "\n")
