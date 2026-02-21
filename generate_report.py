"""
Script para generar reporte HTML de análisis del pipeline de clustering.

Procesa todas las imágenes del dataset y genera:
1. Tabla resumen de cada imagen con métricas
2. Visualización de mejor rama para cada imagen
3. Análisis comparativo y conclusiones

Uso:
    python generate_report.py \
        --json "path/to/_annotations.coco.json" \
        --images_dir "path/to/images" \
        --output "report.html" \
        [--n_steps 4] \
        [--model biomedclip] \
        [--preprocessing clahe]
"""

import argparse
import os
import sys
import json
import base64
import io
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# Configurar matplotlib para usar backend no-interactivo (necesario para scripts)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Importar módulos del proyecto
from utils.coco_loader import list_coco_images
from process_image import ImageProcessingPipeline


def image_to_base64(fig):
    """Convierte una figura matplotlib a base64 para embeber en HTML."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"


def process_single_image(pipeline, image_index):
    """Procesa una imagen individual y retorna resultados y visualización."""
    try:
        tree = pipeline.run(image_index)
        
        if tree is None:
            return None
        
        # Obtener métricas
        best_node = tree.get_best_node()
        optimal_steps = tree.get_optimal_steps()
        leaves = tree.root.get_all_leaves()
        leaves_with_metrics = [l for l in leaves if l.metrics is not None]
        
        if not leaves_with_metrics:
            return None
        
        best_leaf = max(leaves_with_metrics, key=lambda x: x.metrics['f1_coverage'])
        
        # Crear visualización
        fig, ax = plt.subplots(figsize=(14, 12))
        
        if pipeline.img.ndim == 2:
            ax.imshow(pipeline.img, cmap='gray')
        else:
            ax.imshow(pipeline.img)
        
        # Re-calcular limpieza y grouping para mejor nodo
        from utils.evaluation import (
            limpiar_patches_por_componentes_mask,
            agrupar_patches_en_grupos,
            _to_xyxy,
            _overlap_area
        )
        
        kept, removed, _ = limpiar_patches_por_componentes_mask(
            pipeline.img, best_node.patches, min_patches=3, dilate_px=32
        )
        
        grupos, _ = agrupar_patches_en_grupos(
            pipeline.img, kept, min_patches_por_grupo=3, dilate_px=5
        )
        
        # Dibujar grupos (TP en verde, FP en rojo)
        for i, grupo in enumerate(grupos):
            x1, y1, x2, y2 = grupo['position']
            is_tp = best_node.metrics['pred_hits'][i] if i < len(best_node.metrics['pred_hits']) else False
            color = (0.2, 0.8, 0.2) if is_tp else (0.8, 0.2, 0.2)
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.3
            )
            ax.add_patch(rect)
        
        # Determinar GT cubiertos
        gt_covered = []
        for gt in pipeline.boxes:
            gt_xyxy = _to_xyxy(gt, (pipeline.H, pipeline.W))
            covered = False
            for grupo in grupos:
                pred_box = _to_xyxy(grupo['position'], (pipeline.H, pipeline.W))
                if _overlap_area(pred_box, gt_xyxy) > 0:
                    covered = True
                    break
            gt_covered.append(covered)
        
        # Dibujar GT boxes
        for gt_idx, (cls, x1, y1, x2, y2) in enumerate(pipeline.boxes):
            color = 'blue' if gt_covered[gt_idx] else 'orange'
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=color, facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ax.plot(cx, cy, 'o', markersize=10, color=color,
                   markeredgecolor='yellow', markeredgewidth=2)
        
        # Leyenda
        from matplotlib.lines import Line2D
        legend_elements = [
            Rectangle((0, 0), 1, 1, fc=(0.2, 0.8, 0.2), alpha=0.5, label='TP (Correcto)'),
            Rectangle((0, 0), 1, 1, fc=(0.8, 0.2, 0.2), alpha=0.5, label='FP (Incorrecto)'),
            Rectangle((0, 0), 1, 1, fc='none', edgecolor='blue', linewidth=2,
                     linestyle='--', label='GT cubierto'),
            Rectangle((0, 0), 1, 1, fc='none', edgecolor='orange', linewidth=2,
                     linestyle='--', label='GT no cubierto'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        ax.set_title(
            f"Mejor Nodo: {best_node.node_id} (Paso {best_node.step})\n"
            f"TP={best_node.metrics['groups_TP']} | FP={best_node.metrics['groups_FP']} | "
            f"F1={best_node.metrics['f1_coverage']:.3f}",
            fontsize=13, fontweight='bold'
        )
        ax.axis('off')
        
        image_b64 = image_to_base64(fig)
        
        return {
            'image_index': image_index,
            'filename': pipeline.fname,
            'dimensions': f"{pipeline.W}x{pipeline.H}",
            'gt_count': len(pipeline.boxes),
            'best_node': best_node.node_id,
            'optimal_steps': optimal_steps,
            'best_node_f1': best_node.metrics['f1_coverage'],
            'best_node_tp': best_node.metrics['groups_TP'],
            'best_node_fp': best_node.metrics['groups_FP'],
            'best_node_precision': best_node.metrics['group_precision'],
            'best_node_recall': best_node.metrics['gt_recall_coverage'],
            'best_leaf': best_leaf.node_id,
            'best_leaf_step': best_leaf.step,
            'best_leaf_f1': best_leaf.metrics['f1_coverage'],
            'visualization': image_b64,
            'step_stats': tree.get_metrics_by_step()
        }
    
    except Exception as e:
        print(f"  ❌ Error procesando imagen {image_index}: {e}")
        return None


def generate_html_report(results, output_path, json_path, images_dir, n_steps, model_name, preprocessing_method):
    """Genera reporte HTML con todos los resultados."""
    
    # Calcular estadísticas globales
    successful = [r for r in results if r is not None]
    failed = len(results) - len(successful)
    
    if successful:
        avg_f1 = np.mean([r['best_node_f1'] for r in successful])
        avg_precision = np.mean([r['best_node_precision'] for r in successful])
        avg_recall = np.mean([r['best_node_recall'] for r in successful])
        
        total_tp = sum(r['best_node_tp'] for r in successful)
        total_fp = sum(r['best_node_fp'] for r in successful)
        total_gt = sum(r['gt_count'] for r in successful)
        
        images_needing_fewer_steps = sum(1 for r in successful if r['optimal_steps'] < n_steps)
    else:
        avg_f1 = avg_precision = avg_recall = 0
        total_tp = total_fp = total_gt = 0
        images_needing_fewer_steps = 0
    
    # HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Pipeline Clustering</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .config {{
            background: #f5f5f5;
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .config h2 {{
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #333;
        }}
        
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}
        
        .config-item {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        
        .config-item strong {{
            color: #667eea;
            display: block;
            margin-bottom: 5px;
        }}
        
        .summary {{
            padding: 40px;
            background: white;
        }}
        
        .summary h2 {{
            font-size: 1.8em;
            margin-bottom: 30px;
            color: #333;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }}
        
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .stat-card .label {{
            font-size: 0.95em;
            opacity: 0.9;
        }}
        
        .insights {{
            background: #e8f4f8;
            padding: 25px;
            border-left: 4px solid #0288d1;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        
        .insights h3 {{
            color: #0288d1;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        
        .insights ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .insights li {{
            padding: 8px 0;
            border-bottom: 1px solid rgba(2, 136, 209, 0.2);
        }}
        
        .insights li:before {{
            content: "✓ ";
            color: #4caf50;
            font-weight: bold;
            margin-right: 10px;
        }}
        
        .images-section {{
            padding: 40px;
            background: white;
        }}
        
        .images-section h2 {{
            font-size: 1.8em;
            margin-bottom: 30px;
            color: #333;
        }}
        
        .table-responsive {{
            overflow-x: auto;
            margin-bottom: 40px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        thead {{
            background: #667eea;
            color: white;
        }}
        
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 0.95em;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        tbody tr:hover {{
            background: #f5f5f5;
        }}
        
        tbody tr:nth-child(even) {{
            background: #fafafa;
        }}
        
        .value {{
            font-weight: 600;
            color: #667eea;
        }}
        
        .image-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            margin-bottom: 40px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .image-card-header {{
            background: #f5f5f5;
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .image-card-header h3 {{
            font-size: 1.3em;
            color: #333;
            margin-bottom: 10px;
        }}
        
        .image-card-meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            font-size: 0.9em;
        }}
        
        .image-card-meta strong {{
            color: #667eea;
            display: block;
            margin-bottom: 3px;
        }}
        
        .image-card-body {{
            padding: 20px;
        }}
        
        .image-card img {{
            width: 100%;
            max-width: 1000px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        
        .image-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .metric-box {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }}
        
        .metric-box strong {{
            display: block;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .metric-box span {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }}
        
        .footer {{
            background: #f5f5f5;
            padding: 30px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }}
        
        .footer p {{
            margin-bottom: 8px;
        }}
        
        .timestamp {{
            font-size: 0.9em;
            color: #999;
        }}
        
        .recommendations {{
            background: #fff3e0;
            padding: 20px;
            border-left: 4px solid #ff9800;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        
        .recommendations h3 {{
            color: #ff9800;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        
        .recommendations ul {{
            list-style: none;
        }}
        
        .recommendations li {{
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 152, 0, 0.2);
        }}
        
        .recommendations li:before {{
            content: "→ ";
            color: #ff9800;
            font-weight: bold;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- HEADER -->
        <div class="header">
            <h1>🧬 Reporte del Pipeline de Clustering Adaptativo</h1>
            <p>Análisis de detección de núcleos celulares con árbol de clustering binario</p>
        </div>
        
        <!-- CONFIGURACIÓN -->
        <div class="config">
            <h2>⚙️ Configuración del Análisis</h2>
            <div class="config-grid">
                <div class="config-item">
                    <strong>Modelo</strong>
                    {model_name}
                </div>
                <div class="config-item">
                    <strong>Preprocesamiento</strong>
                    {preprocessing_method.upper()}
                </div>
                <div class="config-item">
                    <strong>Pasos Máximos</strong>
                    {n_steps}
                </div>
                <div class="config-item">
                    <strong>Total de Imágenes</strong>
                    {len(results)}
                </div>
            </div>
        </div>
        
        <!-- RESUMEN -->
        <div class="summary">
            <h2>📊 Resumen General</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="value">{len(successful)}</div>
                    <div class="label">Imágenes Procesadas</div>
                </div>
                <div class="stat-card">
                    <div class="value">{avg_f1:.3f}</div>
                    <div class="label">F1 Promedio</div>
                </div>
                <div class="stat-card">
                    <div class="value">{avg_precision:.3f}</div>
                    <div class="label">Precisión Promedio</div>
                </div>
                <div class="stat-card">
                    <div class="value">{avg_recall:.3f}</div>
                    <div class="label">Recall Promedio</div>
                </div>
                <div class="stat-card">
                    <div class="value">{total_tp}/{total_gt}</div>
                    <div class="label">TP / GT Total</div>
                </div>
                <div class="stat-card">
                    <div class="value">{total_fp}</div>
                    <div class="label">FP Total</div>
                </div>
            </div>
            
            <div class="insights">
                <h3>💡 Insights Principales</h3>
                <ul>
                    <li><strong>Desempeño General:</strong> El pipeline logra detectar {(total_tp/total_gt*100):.1f}% de los núcleos (TP: {total_tp}/{total_gt})</li>
                    <li><strong>Falsos positivos:</strong> En total {total_fp} detecciones incorrectas entre todas las imágenes</li>
                    <li><strong>Precisión:</strong> En promedio {avg_precision:.1%} de las detecciones son correctas</li>
                    <li><strong>Recall:</strong> En promedio {avg_recall:.1%} de los núcleos son encontrados</li>
"""
    
    if images_needing_fewer_steps > 0:
        html_content += f"""                    <li><strong>⚠️ Eficiencia:</strong> {images_needing_fewer_steps}/{len(successful)} imágenes necesitarían menos de {n_steps} pasos</li>"""
    
    html_content += """
                </ul>
            </div>
"""
    
    # Recomendaciones
    if images_needing_fewer_steps > 0:
        html_content += f"""
            <div class="recommendations">
                <h3>🎯 Recomendaciones</h3>
                <ul>
                    <li>Considerar reducir N_STEPS a {n_steps - 1} para acelerar ejecución sin perder calidad</li>
                    <li>{images_needing_fewer_steps} de {len(successful)} imágenes alcanzan su mejor resultado en pasos menores</li>
                    <li>Analizar las imágenes individuales en la sección de reportes para ver pasos óptimos</li>
                </ul>
            </div>
"""
    
    html_content += """
        </div>
        
        <!-- TABLA RESUMEN -->
        <div class="images-section">
            <h2>📋 Resumen por Imagen</h2>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Imagen</th>
                            <th>Dimensiones</th>
                            <th>GT</th>
                            <th>Pasos Óptimos</th>
                            <th>Nodo Mejor</th>
                            <th>TP</th>
                            <th>FP</th>
                            <th>Precisión</th>
                            <th>Recall</th>
                            <th>F1</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for i, result in enumerate(successful, 1):
        html_content += f"""
                        <tr>
                            <td>{i}</td>
                            <td>{result['filename']}</td>
                            <td>{result['dimensions']}</td>
                            <td>{result['gt_count']}</td>
                            <td><span class="value">{result['optimal_steps']}</span></td>
                            <td>{result['best_node']}</td>
                            <td>{result['best_node_tp']}</td>
                            <td>{result['best_node_fp']}</td>
                            <td>{result['best_node_precision']:.3f}</td>
                            <td>{result['best_node_recall']:.3f}</td>
                            <td><span class="value">{result['best_node_f1']:.3f}</span></td>
                        </tr>
"""
    
    html_content += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- DETALLE POR IMAGEN -->
        <div class="images-section">
            <h2>🖼️ Análisis Detallado por Imagen</h2>
"""
    
    for i, result in enumerate(successful, 1):
        html_content += f"""
            <div class="image-card">
                <div class="image-card-header">
                    <h3>Imagen {i}: {result['filename']}</h3>
                    <div class="image-card-meta">
                        <div><strong>Dimensiones</strong>{result['dimensions']}</div>
                        <div><strong>Núcleos GT</strong>{result['gt_count']}</div>
                        <div><strong>Pasos Óptimos</strong>{result['optimal_steps']}/{n_steps}</div>
                        <div><strong>F1 Score</strong><span class="value">{result['best_node_f1']:.3f}</span></div>
                    </div>
                </div>
                <div class="image-card-body">
                    <img src="{result['visualization']}" alt="Visualización de {result['filename']}">
                    
                    <div class="image-metrics">
                        <div class="metric-box">
                            <strong>Mejor Nodo</strong>
                            <span>{result['best_node']} (Paso {result['optimal_steps']})</span>
                        </div>
                        <div class="metric-box">
                            <strong>Verdaderos Positivos (TP)</strong>
                            <span>{result['best_node_tp']}</span>
                        </div>
                        <div class="metric-box">
                            <strong>Falsos Positivos (FP)</strong>
                            <span>{result['best_node_fp']}</span>
                        </div>
                        <div class="metric-box">
                            <strong>Precisión</strong>
                            <span>{result['best_node_precision']:.3f}</span>
                        </div>
                        <div class="metric-box">
                            <strong>Recall</strong>
                            <span>{result['best_node_recall']:.3f}</span>
                        </div>
                        <div class="metric-box">
                            <strong>F1 Score</strong>
                            <span>{result['best_node_f1']:.3f}</span>
                        </div>
                    </div>
                </div>
            </div>
"""
    
    html_content += f"""
        </div>
        
        <!-- FOOTER -->
        <div class="footer">
            <p><strong>Reporte generado automáticamente</strong></p>
            <p class="timestamp">Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="margin-top: 15px; font-size: 0.85em;">
                Dataset: {json_path}<br>
                Imágenes: {images_dir}
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    # Escribir archivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Reporte guardado en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generar reporte HTML del pipeline de clustering para todas las imágenes"
    )
    
    parser.add_argument(
        '--json', type=str, required=True,
        help='Ruta al archivo COCO JSON con anotaciones'
    )
    parser.add_argument(
        '--images_dir', type=str, required=True,
        help='Directorio con las imágenes'
    )
    parser.add_argument(
        '--output', type=str, default='reporte_clustering.html',
        help='Archivo de salida del reporte (por defecto: reporte_clustering.html)'
    )
    parser.add_argument(
        '--n_steps', type=int, default=4,
        help='Número de pasos para crecer el árbol (por defecto: 4)'
    )
    parser.add_argument(
        '--model', type=str, default='biomedclip',
        choices=['biomedclip', 'uni', 'optimus', 'uni2'],
        help='Modelo de embeddings a usar (por defecto: biomedclip)'
    )
    parser.add_argument(
        '--preprocessing', type=str, default='clahe',
        choices=['clahe', 'equalize', 'normalize', 'none'],
        help='Método de preprocesamiento (por defecto: clahe)'
    )
    
    args = parser.parse_args()
    
    # Validar rutas
    if not os.path.exists(args.json):
        print(f"❌ Error: Archivo JSON no encontrado: {args.json}")
        sys.exit(1)
    
    if not os.path.isdir(args.images_dir):
        print(f"❌ Error: Directorio no encontrado: {args.images_dir}")
        sys.exit(1)
    
    # Obtener lista de imágenes
    images_list = list_coco_images(args.json)
    print(f"\n📂 Total de imágenes en dataset: {len(images_list)}")
    
    # Crear pipeline
    pipeline = ImageProcessingPipeline(
        json_path=args.json,
        images_dir=args.images_dir,
        model_name=args.model,
        preprocessing_method=args.preprocessing,
        visualize=False,  # No mostrar gráficos intermedios
        n_steps=args.n_steps
    )
    
    # Procesar todas las imágenes
    print(f"\n⏳ Procesando {len(images_list)} imágenes...")
    print("="*70)
    
    results = []
    for i, image_info in enumerate(images_list):
        print(f"\n[{i+1}/{len(images_list)}] Procesando imagen...")
        result = process_single_image(pipeline, i)
        results.append(result)
        
        if result:
            print(f"      ✅ F1: {result['best_node_f1']:.3f} | Pasos óptimos: {result['optimal_steps']}")
        else:
            print(f"      ⚠️ No se pudo procesar")
    
    # Generar reporte
    print("\n" + "="*70)
    print("📄 Generando reporte HTML...")
    generate_html_report(
        results, 
        args.output,
        args.json,
        args.images_dir,
        args.n_steps,
        args.model,
        args.preprocessing
    )
    
    print("\n" + "="*70)
    print("✅ ¡Reporte completado exitosamente!")


if __name__ == '__main__':
    main()
