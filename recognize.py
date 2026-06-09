"""
Script para reconocer caras en una imagen.

Uso:
    python recognize.py <ruta_imagen>

Ejemplo:
    python recognize.py test.jpg
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
    
    print("📥 Descargando Haar Cascade...")
    cascade_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_file)
        print("✓ Haar Cascade descargado\n")
        return True
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False


if not ensure_cascades():
    print("⚠️  No se pudieron descargar los Haar Cascades")
    sys.exit(1)

import cv2
from src.face_recognizer import FaceRecognizer
from src.utils import load_image, save_image
from src.config import RESULTS_DIR


def draw_results(image, results):
    """
    Dibuja los resultados del reconocimiento en la imagen.
    
    Args:
        image: Array de la imagen en RGB
        results: Dict con resultados del reconocimiento
        
    Returns:
        Imagen modificada
    """
    for face_info in results['faces']:
        top = face_info['location']['top']
        right = face_info['location']['right']
        bottom = face_info['location']['bottom']
        left = face_info['location']['left']
        
        name = face_info['name']
        confidence = face_info['confidence']
        
        # Color según si fue reconocido
        if name == "Desconocido":
            color = (255, 0, 0)  # Rojo
            label = "Desconocido"
        else:
            color = (0, 255, 0)  # Verde
            label = f"{name} ({confidence:.2%})"
        
        # Dibujar rectángulo
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        
        # Dibujar etiqueta
        label_y = top - 10 if top > 30 else bottom + 20
        cv2.putText(image, label, (left, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return image


def main():
    if len(sys.argv) < 2:
        print("Uso: python recognize.py <ruta_imagen>")
        print("Ejemplo: python recognize.py test.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Validar que el archivo exista
    if not os.path.exists(image_path):
        print(f"❌ Error: Archivo no encontrado: {image_path}")
        sys.exit(1)
    
    print(f"🔍 Procesando imagen: {image_path}\n")
    
    # Crear reconocedor
    recognizer = FaceRecognizer()
    
    # Cargar imagen
    image = load_image(image_path)
    if image is None:
        print("❌ Error: No se pudo cargar la imagen")
        sys.exit(1)
    
    # Reconocer caras
    results = recognizer.recognize_faces(image)
    
    # Mostrar resultados
    print(f"📊 Resultados:")
    print(f"   • Caras detectadas: {results['faces_detected']}")
    
    if 'error' in results:
        print(f"   • Error: {results['error']}")
    else:
        for i, face in enumerate(results['faces'], 1):
            print(f"\n   Cara {i}:")
            print(f"      • Nombre: {face['name']}")
            print(f"      • Confianza: {face['confidence']:.2%}")
            print(f"      • Posición: top={face['location']['top']}, "
                  f"right={face['location']['right']}, "
                  f"bottom={face['location']['bottom']}, "
                  f"left={face['location']['left']}")
    
    # Guardar imagen con resultados
    if results['faces_detected'] > 0:
        image_with_boxes = draw_results(image.copy(), results)
        output_path = os.path.join(RESULTS_DIR, 
                                   f"result_{os.path.basename(image_path)}")
        save_image(output_path, image_with_boxes)
        print(f"\n✓ Imagen con resultados guardada en: {output_path}")


if __name__ == "__main__":
    main()
