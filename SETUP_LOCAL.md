# Setup Local - Ambiente de Desarrollo

## Estado Actual ✅

✅ **Ambiente virtual creado**: `./venv`  
✅ **Dependencias instaladas**: Todas las librerías de `requirements.txt` + Jupyter  
✅ **Python**: 3.12.2 en el venv  

## Cómo Usar

### 1. Activar el Ambiente (necesario cada vez que abres una terminal)

**PowerShell (Windows):**
```powershell
.\venv\Scripts\Activate.ps1
```

**CMD (Windows):**
```cmd
venv\Scripts\activate.bat
```

**Bash (macOS/Linux):**
```bash
source venv/bin/activate
```

Verás que el prompt cambia a `(venv) PS C:\...>` indicando que el venv está activo.

### 2. Verificar que todo funciona

```bash
python quick_verify.py
```

Debe mostrar `[OK]` para todas las librerías.

### 3. Lanzar Jupyter

```bash
jupyter notebook kaggle_notebook.ipynb
```

Se abrirá en el navegador por defecto. Si prefieres JupyterLab:
```bash
jupyter lab kaggle_notebook.ipynb
```

---

## Cambios Necesarios para Usar Datos Locales

Si quieres correr la notebook con datos locales en lugar de desde Kaggle:

### Opción A: Descargar dataset desde Kaggle

```bash
pip install kagglehub
kaggle datasets download -d "nombre-dataset"
unzip nombre-dataset.zip
```

### Opción B: Editar rutas en la Celda 3

En `kaggle_notebook.ipynb`, Celda 3, cambia:
```python
JSON_PATH = '/kaggle/input/cric-dataset/classifications.json'
BASE_PATH = '/kaggle/input/cric-dataset'
```

Por tus rutas locales, ej:
```python
JSON_PATH = './data/classifications.json'
BASE_PATH = './data'
```

---

## Estructura de Archivos Creados

```
├── venv/                    # Ambiente virtual (ignorar)
├── quick_verify.py          # Script de verificación (ejecutar si hay dudas)
├── LOCAL_SETUP.md           # Esta guía detallada
├── verify_environment.py    # Script de verificación detallado
├── .env.local               # Archivo de configuración (editable)
└── [otros archivos originales]
```

---

## Troubleshooting

### Error: "El archivo no se puede ejecutar porque no existe en el sistema"
**Problema**: No activaste el venv  
**Solución**: Ejecuta primero `.\venv\Scripts\Activate.ps1`

### Error: "No such file or directory: '/kaggle/input...'"
**Problema**: La notebook intenta acceder a datos de Kaggle  
**Solución**: Descarga datos locales o edita rutas como se explica arriba

### Error: "ModuleNotFoundError: No module named 'utils'"
**Problema**: Ejecutaste desde otro directorio  
**Solución**: Asegúrate de ejecutar desde la raíz del proyecto:
```bash
cd c:\Users\mngra\projects\AI\repos\cells-finder-unsupervised
jupyter notebook kaggle_notebook.ipynb
```

### Error: CUDA/GPU
La notebook detectará automáticamente si hay GPU disponible. Si no la hay, ejecutará en CPU (más lento pero funciona).

---

## Próximos Pasos

1. **Haz cambios locales** en la notebook para debug
2. **Verifica que funcione** correctamente
3. **Sincroniza cambios** en GitHub si es necesario
4. **Sube a Kaggle** cuando esté listo

---

## Notas para Desarrollo

- La notebook actualmente tiene un bug en la lógica de matching de TP/FP
- El fix está en [utils/evaluation.py](utils/evaluation.py) - función `evaluar_grupos_vs_boxes_plus`
- Puedes debuggear localmente antes de subirlo a Kaggle

¡Listo para trabajar! 🚀
