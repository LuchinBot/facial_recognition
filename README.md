# 🎭 Aplicación de Reconocimiento Facial

Una aplicación completa y funcional para reconocer caras usando Python, OpenCV y face_recognition.

## ⚙️ Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

**Nota**: En Windows, si `dlib` causa problemas, puedes usar:
```bash
pip install cmake
pip install dlib
```

### 2. Estructura del proyecto

```
facial_recognition/
├── data/
│   ├── known_faces/          # Tus fotos de referencia
│   └── face_encodings/       # Encodings generados (no tocar)
├── src/
│   ├── config.py             # Configuración
│   ├── face_recognizer.py    # Clase principal
│   └── utils.py              # Funciones auxiliares
├── results/                  # Imágenes con resultados
├── main.py                   # Menú principal
├── train.py                  # Script para entrenar
├── recognize.py              # Script para reconocer en imagen
└── requirements.txt
```

## 🚀 Uso

### Opción 1: Menú Interactivo (Recomendado)

```bash
python main.py
```

Esto abrirá un menú con opciones para:
- Entrenar el modelo
- Reconocer caras en imágenes
- Ver estadísticas
- Ver instrucciones

### Opción 2: Entrenar directamente

```bash
python train.py
```

### Opción 3: Reconocer en una imagen

```bash
python recognize.py ruta/imagen.jpg
```

## 📁 Cómo agregar datos

### Paso 1: Estructura de carpetas

Crea carpetas para cada persona en `data/known_faces/`:

```
data/known_faces/
├── Juan/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── foto3.jpg
├── Maria/
│   ├── foto1.jpg
│   └── foto2.jpg
└── Pedro/
    ├── foto1.jpg
    └── foto2.jpg
```

### Paso 2: Agregar fotos

- Coloca 2-3 fotos de cada persona
- Las fotos deben mostrar claramente la cara
- Usa formatos: JPG, PNG, BMP, GIF

### Paso 3: Entrenar el modelo

```bash
python train.py
```

Verás un resumen como:
```
==================================================
📊 Entrenamiento completado:
   • Personas: 3
   • Caras detectadas: 8
   • Encodings generados: 8
==================================================
```

## 🔍 Usar la aplicación

Después del entrenamiento, puedes:

1. **Reconocer en una imagen**:
```bash
python recognize.py test.jpg
```

2. **Usar el menú interactivo**:
```bash
python main.py
```

## ⚙️ Configuración

Edita `src/config.py` para ajustar:

- **FACE_RECOGNITION_TOLERANCE** (0-1):
  - Valores bajos = más estricto (menos falsos positivos)
  - Valores altos = más permisivo (más falsos positivos)
  - Recomendado: 0.6

- **MODEL**:
  - 'hog' = más rápido (recomendado)
  - 'cnn' = más preciso pero lento (requiere GPU)

## 💡 Recomendaciones

✅ **Haz:**
- Usa fotos con buena iluminación
- Múltiples ángulos de la cara
- Diferentes expresiones faciales
- Mínimo 2-3 fotos por persona

❌ **No hagas:**
- Fotos borrosas
- Fotos muy pequeñas
- Caras parcialmente oscurecidas
- Muchas personas en una foto

## 📊 Resultados

Los resultados (imágenes con caras detectadas) se guardan en `results/`.

El formato es: `result_nombre_original_imagen.jpg`

## 🛠️ Solución de problemas

### Error: "No se detectó cara en..."
- La foto es borrosa o muy pequeña
- La cara está de lado o de espaldas
- Prueba con otra foto

### Error: importar dlib
```bash
pip install --upgrade dlib
```

### Modelo muy lento
Cambia `MODEL = 'hog'` en `src/config.py`

### Muchos falsos positivos
Disminuye `FACE_RECOGNITION_TOLERANCE` a 0.5

## 📝 Licencia

Este proyecto usa `face_recognition` (MIT License)

## 🎯 Características

✓ Entrenar con múltiples personas
✓ Reconocer caras en imágenes
✓ Calcular confianza de reconocimiento
✓ Dibujar resultados en imágenes
✓ Menú interactivo fácil de usar
✓ Estructura modular y escalable
✓ Manejo de errores robusto

---

¡Listo para empezar! 🚀
