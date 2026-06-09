"""
Cámara LITE - Versión ultra-rápida para testing.
Solo detección de caras sin reconocimiento.

Uso:
    python camera_lite.py
"""

import sys
import os
from pathlib import Path


def ensure_cascades():
    """Asegura que los Haar Cascades estén disponibles."""
    cascade_dir = Path("data") / "cascades"
    cascade_file = cascade_dir / "haarcascade_frontalface_default.xml"
    
    if cascade_file.exists():
        return str(cascade_file)
    
    print("📥 Descargando Haar Cascade...")
    cascade_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_file)
        print("✓ Haar Cascade descargado\n")
        return str(cascade_file)
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return None


import cv2
import numpy as np
from datetime import datetime
from src.config import RESULTS_DIR


def main():
    print("="*60)
    print("🎥 DETECCIÓN DE CARAS - LITE (ULTRA RÁPIDO)".center(60))
    print("="*60 + "\n")
    
    # Obtener cascade
    cascade_path = ensure_cascades()
    if cascade_path is None:
        print("✗ No se pudo obtener Haar Cascade")
        return
    
    # Cargar cascade
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("✗ Error al cargar cascade")
        return
    
    print("🎥 Abriendo cámara...\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Error: No se pudo abrir la cámara")
        return
    
    # Configurar para máximo rendimiento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✓ Cámara lista\n")
    print("Controles:")
    print("  • Q o ESC: Salir")
    print("  • SPACE: Pausa")
    print("  • S: Guardar captura\n")
    print("="*60 + "\n")
    
    frame_count = 0
    fps = 0
    fps_start = cv2.getTickCount()
    paused = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Detección de caras (rápido)
            if not paused:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Dibujar caras
                for (x, y, w_face, h_face) in faces:
                    cv2.rectangle(frame, (x, y), (x+w_face, y+h_face), (0, 255, 0), 2)
                    cv2.putText(frame, f"Cara detectada", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps_end = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (fps_end - fps_start) * 30
                fps_start = fps_end
            
            # Info
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if paused:
                cv2.putText(frame, "PAUSADO", (w - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(frame, "Q: Salir | SPACE: Pausa | S: Guardar", 
                       (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("🎥 Detección Rápida de Caras", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n✓ Saliendo...")
                break
            elif key == ord(' '):
                paused = not paused
                print("⏸  Pausado" if paused else "▶️  Reanudado")
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(RESULTS_DIR, f"captura_lite_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"💾 Guardado: {filename}")
    
    except KeyboardInterrupt:
        print("\n✓ Interrupción")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Cámara cerrada")


if __name__ == "__main__":
    main()
