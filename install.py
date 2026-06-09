"""
Script de instalación asistida para facial_recognition
Resuelve problemas de instalación en Windows
"""

import subprocess
import sys


def run_command(cmd, description):
    """Ejecuta un comando y muestra el resultado."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"✓ {description} completado exitosamente")
        return True
    else:
        print(f"✗ Error en {description}")
        return False


def main():
    print(f"\n{'='*60}")
    print("🚀 INSTALADOR DE FACIAL RECOGNITION".center(60))
    print(f"{'='*60}")
    
    # Actualizar pip
    print("\n1️⃣  Actualizando pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", 
                "Actualización de pip")
    
    # Instalar cmake (necesario para dlib)
    print("\n2️⃣  Instalando cmake...")
    run_command("pip install cmake", "Instalación de cmake")
    
    # Instalar las dependencias principales
    print("\n3️⃣  Instalando dependencias principales...")
    deps = [
        "numpy",
        "opencv-python",
        "Pillow",
        "scipy",
        "face-recognition"
    ]
    
    for dep in deps:
        print(f"\n   Instalando {dep}...")
        run_command(f"pip install {dep}", f"Instalación de {dep}")
    
    print("\n\n" + "="*60)
    print("✓ ¡Instalación completada!".center(60))
    print("="*60)
    
    # Verificar instalación
    print("\nVerificando instalación...")
    try:
        import face_recognition
        import cv2
        import numpy
        print("✓ Todas las librerías se importan correctamente")
        print("\n✅ Estás listo para usar la aplicación:")
        print("   • python main.py          (menú interactivo)")
        print("   • python train.py         (entrenar modelo)")
        print("   • python recognize.py img.jpg (reconocer en imagen)")
    except ImportError as e:
        print(f"⚠️  Error: {e}")
        print("\nPara instalar manualmente:")
        print("  pip install face-recognition opencv-python Pillow scipy")


if __name__ == "__main__":
    main()
