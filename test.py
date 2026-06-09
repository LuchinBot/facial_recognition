"""
Script de prueba simple para verificar que todo funciona
Ejecuta: python test.py
"""

import sys
import os
from src.face_recognizer import FaceRecognizer
from src.config import KNOWN_FACES_DIR, RESULTS_DIR


def test_installation():
    """Verifica que todas las librerías están instaladas."""
    print("🔍 Verificando instalación...\n")
    
    try:
        import mediapipe
        print("✓ MediaPipe instalado")
    except ImportError:
        print("✗ Error: MediaPipe no instalado")
        return False
    
    try:
        import cv2
        print("✓ OpenCV instalado")
    except ImportError:
        print("✗ Error: OpenCV no instalado")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn instalado")
    except ImportError:
        print("✗ Error: Scikit-learn no instalado")
        return False
    
    try:
        from src.face_recognizer import FaceRecognizer
        print("✓ Módulo face_recognizer funciona\n")
    except ImportError as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def test_folders():
    """Verifica que las carpetas existan."""
    print("📁 Verificando estructura de carpetas...\n")
    
    if os.path.exists(KNOWN_FACES_DIR):
        print(f"✓ Carpeta data/known_faces existe")
        
        # Ver si hay personas
        people = [d for d in os.listdir(KNOWN_FACES_DIR) 
                 if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]
        
        if people:
            print(f"✓ Se encontraron {len(people)} persona(s):")
            for person in people:
                person_dir = os.path.join(KNOWN_FACES_DIR, person)
                images = len([f for f in os.listdir(person_dir) 
                             if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                print(f"   • {person}: {images} imagen(es)")
        else:
            print(f"⚠️  Carpeta data/known_faces está vacía")
            print(f"   📝 Crea carpetas para agregar personas:")
            print(f"      {KNOWN_FACES_DIR}\\Juan\\")
            print(f"      {KNOWN_FACES_DIR}\\Maria\\")
            print(f"      etc...")
    else:
        print(f"✗ Carpeta no existe: {KNOWN_FACES_DIR}")
        return False
    
    if os.path.exists(RESULTS_DIR):
        print(f"✓ Carpeta results existe\n")
    else:
        print(f"⚠️  Carpeta results será creada al generar resultados\n")
    
    return True


def test_model():
    """Intenta crear una instancia del modelo."""
    print("🤖 Verificando modelo...\n")
    
    try:
        recognizer = FaceRecognizer()
        print("✓ Modelo inicializado correctamente")
        
        stats = recognizer.get_stats()
        print(f"\n📊 Estado del modelo:")
        print(f"   • Descriptores cargados: {stats.get('total_descriptors', stats.get('total_embeddings', 0))}")
        print(f"   • Personas conocidas: {stats['total_people']}")
        
        if stats['known_people']:
            print(f"   • Personas: {', '.join(stats['known_people'])}")
        else:
            print(f"   • ⚠️  Sin personas. Ejecuta: python train.py")
        
        return True
    
    except Exception as e:
        print(f"✗ Error al inicializar modelo: {e}")
        return False


def main():
    print("="*60)
    print("🧪 PRUEBA DE INSTALACIÓN Y CONFIGURACIÓN".center(60))
    print("="*60 + "\n")
    
    all_ok = True
    
    # Prueba 1: Instalación
    if not test_installation():
        all_ok = False
    
    # Prueba 2: Carpetas
    if not test_folders():
        all_ok = False
    
    # Prueba 3: Modelo
    if not test_model():
        all_ok = False
    
    print("\n" + "="*60)
    
    if all_ok:
        print("✅ TODO ESTÁ CONFIGURADO CORRECTAMENTE".center(60))
        print("="*60)
        print("\n📝 Próximos pasos:")
        print("   1. Coloca fotos en: data/known_faces/nombre_persona/")
        print("   2. Ejecuta: python train.py")
        print("   3. Ejecuta: python main.py")
    else:
        print("⚠️  HAY PROBLEMAS DE CONFIGURACIÓN".center(60))
        print("="*60)
        print("\n📝 Por favor verifica los errores anteriores.")
    
    print("\n")


if __name__ == "__main__":
    main()
