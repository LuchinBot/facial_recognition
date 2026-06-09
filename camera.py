"""
Reconocimiento facial en tiempo real con cámara web.

Uso:
    python camera.py

Controles:
    - Q o ESC: Salir
    - SPACE: Pausa/Reanuda
    - S: Guardar captura
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
import numpy as np
from collections import deque, Counter
from datetime import datetime
from src.face_recognizer import FaceRecognizer
from src.config import RESULTS_DIR


class PredictionSmoother:
    """
    Suaviza predicciones de reconocimiento usando votación por mayoría.
    Evita que el nombre parpadee entre frames al requerir consistencia.
    """

    def __init__(self, buffer_size=7):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, name, confidence):
        self.buffer.append((name, confidence))

    def get_stable(self):
        """Retorna (name, confidence) solo si hay mayoría clara, si no ('Desconocido', 0)."""
        if not self.buffer:
            return "Desconocido", 0.0
        names = [n for n, _ in self.buffer]
        most_common_name, count = Counter(names).most_common(1)[0]
        if count > len(self.buffer) * 0.5:
            avg_conf = float(np.mean([c for n, c in self.buffer if n == most_common_name]))
            return most_common_name, avg_conf
        return "Desconocido", 0.0

    def reset(self):
        self.buffer.clear()


class CameraRecognizer:
    """Reconocimiento facial en tiempo real con cámara web."""

    def __init__(self):
        self.recognizer = FaceRecognizer()
        self.cap = None
        self.paused = False
        self.skip_frames = 2  # procesar cada 2 frames para balance velocidad/precisión
        self.frame_count = 0
        # Suavizadores por slot de cara (hasta 4 caras simultáneas)
        self.smoothers = [PredictionSmoother() for _ in range(4)]

        if not self.recognizer.known_face_descriptors:
            print("⚠️  No hay modelo entrenado")
            print("Ejecuta: python train.py")
            sys.exit(1)

        stats = self.recognizer.get_stats()
        print(f"✓ Modelo cargado: {stats['total_people']} persona(s)")
        print(f"✓ Personas: {', '.join(stats['known_people'])}")
        print(f"✓ Método: {stats['model']}\n")
    
    def draw_results(self, frame, smoothed_faces):
        """
        Dibuja resultados suavizados sobre el frame.
        smoothed_faces: lista de dicts con location, name, confidence (ya suavizados).
        """
        for face_info in smoothed_faces:
            top = face_info['location']['top']
            right = face_info['location']['right']
            bottom = face_info['location']['bottom']
            left = face_info['location']['left']

            name = face_info['name']
            confidence = face_info['confidence']

            if name == "Desconocido":
                color = (0, 0, 255)
                label = "Desconocido"
            else:
                color = (0, 220, 0)
                label = f"{name}  {confidence:.0%}"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_y = top - 12 if top > 30 else bottom + label_size[1] + 12
            bg_top = label_y - label_size[1] - 6
            bg_bot = label_y + baseline + 2
            cv2.rectangle(frame, (left, bg_top), (left + label_size[0] + 8, bg_bot), color, -1)
            cv2.putText(frame, label, (left + 4, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame
    
    def draw_info(self, frame, fps, paused):
        """Dibuja información en el frame."""
        h, w = frame.shape[:2]
        
        # Información superior izquierda
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Estado de pausa
        if paused:
            cv2.putText(frame, "PAUSADO", (w - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Instrucciones
        cv2.putText(frame, "Q: Salir | SPACE: Pausa | S: Guardar", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Inicia el reconocimiento en tiempo real - Versión rápida."""
        print("🎥 Abriendo cámara (esto puede tardar unos segundos)...\n")
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("✗ Error: No se pudo abrir la cámara")
            print("Verifica que la cámara esté conectada y disponible")
            return
        
        # Configurar para máximo rendimiento
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # Resolución baja para velocidad
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # Buffer pequeño
        
        # Descartar primeros frames para permitir que se estabilice
        print("⏳ Estabilizando cámara...")
        for _ in range(10):
            self.cap.read()
        
        print("✓ Cámara lista\n")
        print("Presiona Q o ESC para salir | SPACE para pausar | S para guardar\n")
        print("="*60 + "\n")
        
        frame_count = 0
        fps_start = cv2.getTickCount()
        fps = 0
        last_raw_results = None
        last_smoothed_faces = []

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    print("✗ Error al capturar frame")
                    break

                frame = cv2.flip(frame, 1)

                # Reconocer caras cada N frames
                if not self.paused and frame_count % self.skip_frames == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_raw_results = self.recognizer.recognize_faces(rgb_frame)

                    # Alimentar suavizadores con nuevas predicciones
                    if last_raw_results and 'faces' in last_raw_results:
                        raw_faces = last_raw_results['faces']
                        # Ordenar caras por posición X para asignar slots consistentemente
                        raw_faces_sorted = sorted(raw_faces, key=lambda f: f['location']['left'])

                        # Resetear slots que ya no tienen cara
                        for i in range(len(raw_faces_sorted), len(self.smoothers)):
                            self.smoothers[i].reset()

                        # Actualizar slots activos
                        last_smoothed_faces = []
                        for i, face_info in enumerate(raw_faces_sorted[:len(self.smoothers)]):
                            self.smoothers[i].add(face_info['name'], face_info['confidence'])
                            stable_name, stable_conf = self.smoothers[i].get_stable()
                            last_smoothed_faces.append({
                                'name': stable_name,
                                'confidence': stable_conf,
                                'location': face_info['location']
                            })
                    else:
                        for s in self.smoothers:
                            s.reset()
                        last_smoothed_faces = []

                # Dibujar con resultados suavizados
                if last_smoothed_faces:
                    frame = self.draw_results(frame, last_smoothed_faces)
                
                # Calcular FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps_end = cv2.getTickCount()
                    fps = cv2.getTickFrequency() / (fps_end - fps_start) * 30
                    fps_start = fps_end
                
                # Dibujar información
                frame = self.draw_info(frame, fps, self.paused)
                
                # Mostrar frame
                cv2.imshow("🎭 Reconocimiento Facial en Tiempo Real", frame)
                
                # Manejar teclas (sin bloqueo)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q o ESC
                    print("\n✓ Saliendo...")
                    break
                
                elif key == ord(' '):  # SPACE - Pausa
                    self.paused = not self.paused
                    if self.paused:
                        print("⏸  Pausado")
                    else:
                        print("▶️  Reanudado")
                
                elif key == ord('s'):  # S - Guardar captura
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(RESULTS_DIR, f"captura_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"💾 Guardado: {filename}")
        
        except KeyboardInterrupt:
            print("\n✓ Interrupción del usuario")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("✓ Cámara cerrada")


def main():
    print("="*60)
    print("🎥 RECONOCIMIENTO FACIAL EN TIEMPO REAL".center(60))
    print("="*60 + "\n")
    
    try:
        recognizer = CameraRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
