"""
Script para entrenar el modelo de reconocimiento facial.

Uso:
    python train.py

Estructura esperada:
    data/known_faces/
        ├── persona_1/
        │   ├── foto1.jpg
        │   ├── foto2.jpg
        │   └── ...
        ├── persona_2/
        │   ├── foto1.jpg
        │   └── ...
        └── ...
"""

import sys
import os
from pathlib import Path


def ensure_cascades():
    """Asegura que los Haar Cascades estén disponibles."""
    cascade_dir = Path("data") / "cascades"
    cascade_file = cascade_dir / "haarcascade_frontalface_default.xml"
    
    if cascade_file.exists():
        return True
    
    print("📥 Descargando Haar Cascade necesarios...")
    cascade_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_file)
        print(f"✓ Haar Cascade descargado\n")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


from src.face_recognizer import FaceRecognizer


def main():
    print("🚀 Iniciando entrenamiento de reconocimiento facial...\n")
    
    # Asegurar que los cascades estén disponibles
    if not ensure_cascades():
        print("⚠️  Intenta descargar manualmente desde:")
        print("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
        sys.exit(1)
    
    # Crear instancia del reconocedor
    recognizer = FaceRecognizer()
    
    # Entrenar
    stats = recognizer.train(verbose=True)
    
    # Mostrar resumen
    if stats['total_encodings'] > 0:
        print("\n✅ ¡Entrenamiento exitoso!")
        print(f"\nResumen:")
        model_stats = recognizer.get_stats()
        print(f"  • Personas reconocidas: {model_stats['total_people']}")
        print(f"  • Descriptores totales: {model_stats['total_descriptors']}")
        print(f"  • Personas: {', '.join(model_stats['known_people'])}")
    else:
        print("\n⚠️  No se generaron descriptores. Verifica la estructura de carpetas.")
        if stats['errors']:
            print("\nErrores encontrados:")
            for error in stats['errors'][:3]:
                print(f"  • {error}")


if __name__ == "__main__":
    main()
