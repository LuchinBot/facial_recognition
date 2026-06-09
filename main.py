"""
Aplicación principal de reconocimiento facial.
Menú interactivo para entrenar y reconocer caras.
"""

import os
import sys
import subprocess
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

from src.face_recognizer import FaceRecognizer
from src.utils import load_image, save_image
from src.config import KNOWN_FACES_DIR, RESULTS_DIR
import cv2


def clear_screen():
    """Limpia la pantalla según el SO."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_menu():
    """Imprime el menú principal."""
    clear_screen()
    print("=" * 60)
    print("🎭 APLICACIÓN DE RECONOCIMIENTO FACIAL".center(60))
    print("=" * 60)
    print("\n1. Entrenar modelo (cargar nuevas caras)")
    print("2. Reconocer caras en una imagen")
    print("3. 🎥 Reconocer caras en tiempo real (CÁMARA)")
    print("4. Ver estadísticas del modelo")
    print("5. Mostrar instrucciones para agregar datos")
    print("6. Salir")
    print("\n" + "=" * 60)


def train_model():
    """Entrena el modelo."""
    print("\n📚 Entrenando modelo...\n")
    recognizer = FaceRecognizer()
    stats = recognizer.train(verbose=True)
    
    if stats['total_encodings'] == 0:
        print("\n⚠️  No se encontraron caras. ¿Agregaste imágenes?")
        print("Ver opción 5 para instrucciones.")
    
    input("\nPresiona Enter para continuar...")


def recognize_image():
    """Reconoce caras en una imagen."""
    print("\n🔍 Reconocer caras en imagen")
    print("-" * 40)
    
    image_path = input("Ingresa la ruta de la imagen: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ Archivo no encontrado: {image_path}")
        input("Presiona Enter para continuar...")
        return
    
    print("\nProcesando...")
    recognizer = FaceRecognizer()
    image = load_image(image_path)
    
    if image is None:
        print("❌ No se pudo cargar la imagen")
        input("Presiona Enter para continuar...")
        return
    
    results = recognizer.recognize_faces(image)
    
    # Mostrar resultados
    print(f"\n📊 Resultados:")
    print(f"   Caras detectadas: {results['faces_detected']}")
    
    if results['faces_detected'] == 0:
        print("   No se detectaron caras.")
    else:
        for i, face in enumerate(results['faces'], 1):
            print(f"\n   Cara {i}:")
            print(f"      • Nombre: {face['name']}")
            print(f"      • Confianza: {face['confidence']:.2%}")
    
    # Guardar resultado
    if results['faces_detected'] > 0:
        from recognize import draw_results
        image_with_boxes = draw_results(image.copy(), results)
        output_path = os.path.join(RESULTS_DIR, 
                                   f"result_{os.path.basename(image_path)}")
        save_image(output_path, image_with_boxes)
        print(f"\n✓ Imagen guardada en: {output_path}")
    
    input("\nPresiona Enter para continuar...")


def recognize_camera():
    """Abre la cámara para reconocer caras en tiempo real."""
    print("\n🎥 Reconocimiento en Tiempo Real")
    print("-" * 40)
    print("Abriendo cámara...\n")
    
    subprocess.run([sys.executable, "camera.py"])
    
    input("\nPresiona Enter para continuar...")


def show_stats():
    """Muestra estadísticas del modelo."""
    print("\n📊 Estadísticas del Modelo")
    print("-" * 40)
    
    recognizer = FaceRecognizer()
    stats = recognizer.get_stats()
    
    print(f"Total de descriptores: {stats['total_descriptors']}")
    print(f"Personas conocidas: {stats['total_people']}")
    
    if stats['known_people']:
        print("\nPersonas registradas:")
        for person in sorted(stats['known_people']):
            print(f"   • {person}")
    else:
        print("\n⚠️  No hay personas registradas aún.")
    
    print(f"\nTolerancia: {stats['tolerance']}")
    print(f"Modelo: {stats['model']}")
    
    input("\nPresiona Enter para continuar...")


def show_instructions():
    """Muestra instrucciones para agregar datos."""
    clear_screen()
    print("=" * 60)
    print("📋 INSTRUCCIONES PARA AGREGAR DATOS".center(60))
    print("=" * 60)
    
    print(f"""
1. ESTRUCTURA DE CARPETAS
   ├── data/
   │   ├── known_faces/
   │   │   ├── Juan/
   │   │   │   ├── foto1.jpg
   │   │   │   ├── foto2.jpg
   │   │   │   └── foto3.jpg
   │   │   ├── Maria/
   │   │   │   ├── foto1.jpg
   │   │   │   └── foto2.jpg
   │   │   └── Pedro/
   │   │       ├── foto1.jpg
   │   │       └── foto2.jpg
   │   └── face_encodings/

2. ¿CÓMO AGREGAR UNA NUEVA PERSONA?
   
   a) Crea una carpeta en: {KNOWN_FACES_DIR}
      Ejemplo: {KNOWN_FACES_DIR}\\Juan
   
   b) Coloca fotos de esa persona en esa carpeta
      • Mínimo 2-3 fotos recomendadas
      • Formato: JPG, PNG, BMP, GIF
      • Las caras deben ser claras y visibles
   
   c) Ejecuta: python train.py
      Esto generará los descriptores

3. RECOMENDACIONES
   ✓ Usa fotos con buena iluminación
   ✓ Múltiples ángulos de la cara
   ✓ Diferentes expresiones faciales
   ✗ No uses fotos borrosas o pequeñas
   ✗ Evita fotos con varias personas

4. DESPUÉS DE AGREGAR DATOS
   • Ejecuta: python train.py
   • Ejecuta: python main.py
   • Ejecuta: python camera.py (para cámara)

5. CARPETA DE RESULTADOS
   Tus resultados se guardarán en: {RESULTS_DIR}
""")
    
    input("Presiona Enter para continuar...")


def main():
    """Menú principal."""
    recognizer = FaceRecognizer()
    
    while True:
        print_menu()
        
        choice = input("Selecciona una opción (1-6): ").strip()
        
        if choice == '1':
            train_model()
        elif choice == '2':
            recognize_image()
        elif choice == '3':
            recognize_camera()
        elif choice == '4':
            show_stats()
        elif choice == '5':
            show_instructions()
        elif choice == '6':
            print("\n👋 ¡Hasta luego!")
            sys.exit(0)
        else:
            print("\n❌ Opción no válida. Intenta de nuevo.")
            input("Presiona Enter para continuar...")


if __name__ == "__main__":
    main()
