# 🚀 GUÍA DE INICIO RÁPIDO

## ¿Qué necesitas hacer?

### Paso 1: Agregar tus datos (fotos de caras)

```
d:\00 - PROGRAMMING\0 - LOCAL\PYTHON\facial_recognition\data\known_faces\
├── Juan/
│   ├── juan_foto1.jpg
│   ├── juan_foto2.jpg
│   └── juan_foto3.jpg
├── Maria/
│   ├── maria_foto1.jpg
│   └── maria_foto2.jpg
└── Pedro/
    ├── pedro_foto1.jpg
    └── pedro_foto2.jpg
```

**Instrucciones:**
1. Abre la carpeta: `data/known_faces/`
2. Crea una carpeta para cada persona (usa su nombre)
3. Coloca 2-3 fotos de esa persona en su carpeta
4. Las fotos deben mostrar **claramente la cara**

### Paso 2: Entrenar el modelo

Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
python train.py
```

**Verás algo como:**
```
✓ Juan: 3 cara(s)
✓ Maria: 2 cara(s)
✓ Pedro: 2 cara(s)

==================================================
📊 Entrenamiento completado:
   • Personas: 3
   • Caras detectadas: 7
   • Embeddings: 7
==================================================
```

### Paso 3: Reconocer caras

Opción A - Menú interactivo (recomendado):
```bash
python main.py
```

Opción B - Reconocer en una imagen específica:
```bash
python recognize.py ruta/a/tu/imagen.jpg
```

## 📁 Estructura del proyecto

```
facial_recognition/
├── data/
│   ├── known_faces/           ← AQUÍ PONES TUS FOTOS
│   │   ├── Juan/
│   │   ├── Maria/
│   │   └── Pedro/
│   └── face_encodings/        ← Se genera automáticamente
│       ├── encodings.pkl
│       └── names.pkl
├── src/
│   ├── face_recognizer.py     ← Lógica de reconocimiento
│   ├── utils.py               ← Funciones auxiliares
│   ├── config.py              ← Configuración
│   └── __init__.py
├── results/                   ← Donde se guardan resultados
├── main.py                    ← Menú principal
├── train.py                   ← Entrenar modelo
├── recognize.py               ← Reconocer en imagen
├── install.py                 ← Instalar dependencias
└── README.md                  ← Documentación completa
```

## 💡 Consejos importantes

### ✅ Haz esto:
- Usa fotos de **buena calidad**
- Las caras deben estar **claras y visibles**
- **Mínimo 2-3 fotos por persona**
- Diferentes **ángulos**: frontal, 3/4, perfil
- Diferentes **expresiones**: normal, sonriendo
- **Buena iluminación** (luz natural si es posible)

### ❌ Evita esto:
- Fotos borrosas o pixeladas
- Caras muy pequeñas
- Caras de espaldas o de lado extremo
- Muchas personas en una foto
- Fotos en la oscuridad

## 🔧 Comandos útiles

```bash
# Ver menú interactivo
python main.py

# Entrenar el modelo con tus fotos
python train.py

# Reconocer caras en una imagen
python recognize.py imagen.jpg

# Ver la estructura de carpetas
dir data/known_faces

# Limpiar resultados anteriores
del results\*
```

## 📊 Ejemplo de salida

Cuando reconoces caras, ves:
```
📊 Resultados:
   Caras detectadas: 3

   Cara 1:
      • Nombre: Juan
      • Confianza: 95.23%
      • Posición: top=100, right=250, bottom=200, left=150

   Cara 2:
      • Nombre: Maria
      • Confianza: 92.15%

   Cara 3:
      • Nombre: Desconocido
      • Confianza: 0.00%

✓ Imagen guardada en: results/result_imagen.jpg
```

## ⚡ Rendimiento

- **Primera ejecución:** Lenta (primera inicialización de modelos)
- **Siguientes ejecuciones:** Rápidas (3-5 segundos por imagen)
- **Precisión:** ~95% con buenas fotos
- **Velocidad:** Optimizada para Windows

## 🆘 Problemas frecuentes

**P: "No se detectó cara en..."**
R: La foto no es clara. Usa una mejor imagen.

**P: El modelo da falsos positivos**
R: Edita `src/config.py` y baja `FACE_RECOGNITION_TOLERANCE` a 0.5

**P: ¿Cuántas fotos necesito?**
R: Mínimo 2-3 por persona. Más fotos = mejor precisión.

**P: ¿Qué formatos de imagen funcionan?**
R: JPG, PNG, BMP, GIF (cualquier formato estándar)

---

## 📝 Próximos pasos

1. ✅ Crea carpetas en `data/known_faces/`
2. ✅ Agrega tus fotos
3. ✅ Ejecuta `python train.py`
4. ✅ Ejecuta `python main.py` para usar
5. ✅ ¡Listo! 🎉

¿Necesitas ayuda? Revisa `README.md` para documentación completa.
