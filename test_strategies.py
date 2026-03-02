"""
Script de prueba para verificar el funcionamiento de las estrategias de evaluación.
"""
import os
import sys

# Configuración de rutas
JSON_PATH = "./data/cric_searchable_all_ccedd.json"
IMAGES_DIR = "./data/CRIC_CCEDD_ONLY_KERNELS"
CATEGORY_IDS = [4, 5]

def test_segmentation_strategy():
    """Prueba básica de la estrategia de segmentación."""
    print("="*70)
    print("PRUEBA: Estrategia de Segmentación")
    print("="*70)
    
    from process_image import ImageProcessingPipeline
    
    # Crear pipeline con estrategia de segmentación
    pipeline = ImageProcessingPipeline(
        json_path=JSON_PATH,
        images_dir=IMAGES_DIR,
        model_name='biomedclip',
        preprocessing_method='clahe',
        category_ids=CATEGORY_IDS,
        visualize=False,
        n_steps=3,
        evaluation_strategy='segmentation'
    )
    
    print("\n✅ Pipeline creado con estrategia de segmentación")
    
    # Cargar imagen
    pipeline.load_image(0)
    print(f"✅ Imagen cargada: {pipeline.fname}")
    
    # Extraer embeddings
    pipeline.extract_embeddings()
    print(f"✅ Embeddings extraídos: {len(pipeline.patch_data)} patches")
    
    # Construir árbol
    tree = pipeline.build_tree()
    print(f"✅ Árbol construido")
    
    # Obtener mejor nodo
    best = tree.get_best_node()
    if best:
        print(f"\n🏆 Mejor nodo: {best.node_id} (Paso {best.step})")
        print(f"   Métricas:")
        for key, value in best.metrics.items():
            if isinstance(value, float):
                print(f"   - {key}: {value:.4f}")
            elif isinstance(value, int):
                print(f"   - {key}: {value}")
    
    # Visualizar
    print("\n📸 Generando visualización...")
    tree.visualize_best_node_with_grouping()
    
    print("\n✅ PRUEBA EXITOSA: Estrategia de segmentación")
    return tree

def test_bbox_strategy():
    """Prueba básica de la estrategia de bounding boxes."""
    print("\n\n" + "="*70)
    print("PRUEBA: Estrategia de Bounding Boxes (Legacy)")
    print("="*70)
    
    from process_image import ImageProcessingPipeline
    
    # Crear pipeline con estrategia de bbox
    pipeline = ImageProcessingPipeline(
        json_path=JSON_PATH,
        images_dir=IMAGES_DIR,
        model_name='biomedclip',
        preprocessing_method='clahe',
        category_ids=CATEGORY_IDS,
        visualize=False,
        n_steps=3,
        evaluation_strategy='bbox'
    )
    
    print("\n✅ Pipeline creado con estrategia de bbox")
    
    # Cargar imagen
    pipeline.load_image(0)
    print(f"✅ Imagen cargada: {pipeline.fname}")
    
    # Extraer embeddings
    pipeline.extract_embeddings()
    print(f"✅ Embeddings extraídos: {len(pipeline.patch_data)} patches")
    
    # Construir árbol
    tree = pipeline.build_tree()
    print(f"✅ Árbol construido")
    
    # Obtener mejor nodo
    best = tree.get_best_node()
    if best:
        print(f"\n🏆 Mejor nodo: {best.node_id} (Paso {best.step})")
        print(f"   Métricas:")
        for key, value in best.metrics.items():
            if isinstance(value, float):
                print(f"   - {key}: {value:.4f}")
            elif isinstance(value, int):
                print(f"   - {key}: {value}")
    
    # Visualizar
    print("\n📸 Generando visualización...")
    tree.visualize_best_node_with_grouping()
    
    print("\n✅ PRUEBA EXITOSA: Estrategia de bbox")
    return tree

def compare_strategies():
    """Compara ambas estrategias."""
    print("\n\n" + "="*70)
    print("COMPARACIÓN DE ESTRATEGIAS")
    print("="*70)
    
    from process_image import ImageProcessingPipeline
    
    strategies = ['bbox', 'segmentation']
    results = {}
    
    for strategy in strategies:
        print(f"\n🔄 Procesando con estrategia: {strategy}")
        
        pipeline = ImageProcessingPipeline(
            json_path=JSON_PATH,
            images_dir=IMAGES_DIR,
            model_name='biomedclip',
            preprocessing_method='clahe',
            category_ids=CATEGORY_IDS,
            visualize=False,
            n_steps=3,
            evaluation_strategy=strategy
        )
        
        pipeline.load_image(0)
        pipeline.extract_embeddings()
        tree = pipeline.build_tree()
        
        best = tree.get_best_node()
        results[strategy] = {
            'f1': best.metrics['f1_coverage'],
            'metrics': best.metrics
        }
    
    print("\n" + "="*70)
    print("RESULTADOS DE COMPARACIÓN")
    print("="*70)
    
    for strategy, data in results.items():
        print(f"\n{strategy.upper()}:")
        print(f"  F1 Coverage: {data['f1']:.4f}")
        
        if strategy == 'bbox':
            print(f"  Groups TP: {data['metrics']['groups_TP']}")
            print(f"  Groups FP: {data['metrics']['groups_FP']}")
            print(f"  Precision: {data['metrics']['group_precision']:.4f}")
            print(f"  Recall: {data['metrics']['gt_recall_coverage']:.4f}")
        else:
            print(f"  DICE: {data['metrics']['dice']:.4f}")
            print(f"  IoU: {data['metrics']['iou']:.4f}")
            print(f"  Precision: {data['metrics']['precision']:.4f}")
            print(f"  Recall: {data['metrics']['recall']:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Probar estrategias de evaluación")
    parser.add_argument('--mode', type=str, default='segmentation',
                       choices=['segmentation', 'bbox', 'compare'],
                       help='Modo de prueba')
    
    args = parser.parse_args()
    
    if args.mode == 'segmentation':
        test_segmentation_strategy()
    elif args.mode == 'bbox':
        test_bbox_strategy()
    elif args.mode == 'compare':
        compare_strategies()
