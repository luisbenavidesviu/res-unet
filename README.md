# Deep Residual U-Net

## Descripción

Implementación de una arquitectura **Deep Residual U-Net** para segmentación de imágenes, adaptada a partir del repositorio oficial:

**Base:** [DuFanXin/deep_residual_unet](https://github.com/DuFanXin/deep_residual_unet)

Esta versión incluye las funciones principales de construcción del modelo (`model.py`), utilidades para procesamiento de datos (`util.py`) y rutinas de entrenamiento (`train.py`).

---

## Estructura del Proyecto

```
deep_residual_unet_updated/
├── README.md              # Este archivo
├── model.py               # Definición de la arquitectura ResUNet
├── util.py                # Utilidades: iteradores, métricas, aumentación de datos
├── train.py               # Script de entrenamiento
└── bash.py                # Descarga automática del dataset
```

---

## Descarga del Dataset

### `bash.py` - Descargador Automático

El archivo `bash.py` permite descargar automáticamente el dataset completo desde los servidores oficiales de la **Universidad de Massachusetts**.

**Uso:**

```bash
python bash.py
```

Este script descargará:
- Imágenes satelitales (`.tiff`)
- Máscaras de segmentación (`.tif`)
- Divisiones `train/`, `valid/` y `test/`

Todos los datos se organizarán en la carpeta `dataset/` siguiendo la estructura esperada por el iterador `PASCALVOCIterator`.

---

## Componentes Principales

### `model.py`
Define la arquitectura ResUNet con:
- Bloques residuales (`res_block`)
- Encoder con downsampling
- Bottleneck
- Decoder con upsampling y concatenación de características
- Salida final con sigmoide para segmentación binaria

**Función principal:**
```python
from model import build_res_unet

model = build_res_unet(input_shape=(512, 512, 1))
```

### `util.py`
Incluye:
- `PASCALVOCIterator`: Generador de datos con aumentación
- `dice_coef`: Métrica de similitud Dice
- `dice_coef_loss`: Función de pérdida Dice
- Soporte para validación y test

### `train.py`
Script para entrenar el modelo con:
- Configuración de hiperparámetros
- Callbacks de checkpoint y TensorBoard
- Generadores de entrenamiento y validación

---

## Referencia

- **Papel Original:** [Road Extraction by Deep Residual U-Net](https://arxiv.org/pdf/1711.10684)
- **Repositorio Base:** https://github.com/DuFanXin/deep_residual_unet
- **Dataset:** Universidad de Massachusetts (PASCAL VOC / similar)

---

## Requisitos

- TensorFlow/Keras
- NumPy
- jupyter
- pandas
- matplotlib
- opencv-python-headless

Para instalar:
```bash
pip install -r requirements.txt
```
