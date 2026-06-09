# 🎥 Opciones de Cámara

## 🚀 3 opciones disponibles:

### 1. **camera_lite.py** - ⚡ ULTRA RÁPIDO (Solo detección)
```bash
python camera_lite.py
```
✅ Abre **instantáneamente**
✅ Solo detecta dónde están las caras
✅ Perfecto para testing
⏱ ~100 ms por frame

### 2. **camera.py** - 🎭 Reconocimiento Completo
```bash
python camera.py
```
✅ Detecta Y reconoce caras
✅ Muestra nombre y confianza
✅ Procesa cada 3 frames (más rápido)
⏱ ~300-500 ms por frame

### 3. **main.py** - Menú Principal
```bash
python main.py
```
Selecciona opción **3** para cámara

---

## ⌨️ Controles

| Tecla | Función |
|-------|---------|
| **Q** o **ESC** | Salir |
| **SPACE** | Pausar/Reanudar |
| **S** | Guardar captura |

---

## 📊 Comparación

| Feature | Lite | Full |
|---------|------|------|
| Detección | ✅ | ✅ |
| Reconocimiento | ❌ | ✅ |
| Velocidad | ⚡⚡⚡ | ⚡⚡ |
| Latencia | 50-100ms | 300-500ms |
| Nombre | ❌ | ✅ |
| Confianza | ❌ | ✅ |

---

## 🔧 Si la cámara aún es lenta:

### Opción A: Usar camera_lite.py
La versión lite es **3-5x más rápida** porque no hace reconocimiento.

### Opción B: Reducir resolución
Edita `camera.py` línea ~140:
```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Más bajo = más rápido
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
```

### Opción C: Aumentar skip_frames
En `camera.py` línea ~115:
```python
self.skip_frames = 5  # Procesar cada 5 frames en vez de cada 3
```

---

## 🎯 Recomendación

**Para uso normal**: `python camera.py`  
**Para testing rápido**: `python camera_lite.py`  
**Con menú**: `python main.py` → opción 3

---

## 💡 Tips

- Asegúrate que la cámara esté bien conectada
- La primera ejecución puede tardar (carga modelos)
- Las siguientes son más rápidas (caché)
- Si está lento, usa `camera_lite.py` primero
