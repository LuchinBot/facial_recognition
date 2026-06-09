"""
Reconocimiento facial con OpenCV Haar Cascade + LBP + 1-NN
LBP (Local Binary Patterns) es robusto a iluminacion y funciona con pocos ejemplos.
"""

import os
import sys
import pickle
import numpy as np
import cv2
from pathlib import Path
from src.config import KNOWN_FACES_DIR, ENCODINGS_FILE, NAMES_FILE, FACE_RECOGNITION_TOLERANCE
from src.utils import get_image_files, load_image, is_valid_image

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

FACE_SIZE = (100, 100)
LBP_GRID = 7          # grilla 7x7 para LBP espacial
MODEL_FILE = os.path.join(os.path.dirname(ENCODINGS_FILE), 'face_model.pkl')
DEFAULT_THRESHOLD = 0.72   # similitud coseno mínima para identificar


def get_cascade_path():
    local_cascade = Path("data") / "cascades" / "haarcascade_frontalface_default.xml"
    if local_cascade.exists():
        return str(local_cascade)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        return cascade_path
    print("Descargando Haar Cascade...")
    try:
        import urllib.request
        local_cascade.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, local_cascade)
        return str(local_cascade)
    except Exception as e:
        print(f"Error descargando cascade: {e}")
        return None


def _compute_lbp(gray):
    """LBP basico: compara cada pixel con sus 8 vecinos."""
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    g = gray.astype(np.int16)
    for bit, (dy, dx) in enumerate(offsets):
        shifted = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        lbp |= ((shifted >= g).astype(np.uint8) << bit)
    return lbp


def _lbp_features(gray_face):
    """
    Extrae histogramas LBP en grilla NxN.
    Resultado invariante a cambios monotonicos de iluminacion.
    """
    lbp = _compute_lbp(gray_face)
    h, w = lbp.shape
    cell_h = h // LBP_GRID
    cell_w = w // LBP_GRID
    features = []
    for i in range(LBP_GRID):
        for j in range(LBP_GRID):
            cell = lbp[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            hist, _ = np.histogram(cell.ravel(), bins=59, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)
            features.append(hist)
    vec = np.concatenate(features)
    # Normalizar vector final
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def _preprocess_face(face_region):
    """Convierte region de cara a vector LBP normalizado."""
    if face_region is None or face_region.size == 0:
        return None
    gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY) if len(face_region.shape) == 3 else face_region.copy()
    resized = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    # CLAHE para normalizar iluminacion local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(resized)
    return _lbp_features(equalized)


class FaceRecognizer:
    """Reconocedor facial: Haar Cascade para deteccion + LBP + 1-NN para identificacion."""

    def __init__(self, tolerance=FACE_RECOGNITION_TOLERANCE):
        self.tolerance = tolerance
        # Plantillas almacenadas: lista de (nombre, vector_lbp)
        self.face_templates = []
        # Umbral adaptativo calculado durante entrenamiento
        self.threshold = DEFAULT_THRESHOLD

        # Compatibilidad con checks externos (camera.py)
        self.known_face_descriptors = []
        self.known_face_names = []

        cascade_path = get_cascade_path()
        if cascade_path is None:
            raise RuntimeError("No se pudo obtener Haar Cascade")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Error al cargar cascade: {cascade_path}")

        self.load_encodings()

    def _detect_faces(self, image):
        """Detecta caras. Retorna lista de (top, right, bottom, left)."""
        if self.face_cascade.empty():
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        try:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        except Exception:
            return []
        if len(faces) == 0:
            return []
        # Suprimir detecciones superpuestas — quedarse con la mas grande de cada grupo
        faces_list = list(faces)
        faces_list.sort(key=lambda f: f[2] * f[3], reverse=True)
        kept = []
        for (x, y, w, h) in faces_list:
            overlaps = False
            for (kx, ky, kw, kh) in kept:
                # IoU simple
                ix = max(0, min(x + w, kx + kw) - max(x, kx))
                iy = max(0, min(y + h, ky + kh) - max(y, ky))
                if ix * iy > 0.3 * min(w * h, kw * kh):
                    overlaps = True
                    break
            if not overlaps:
                kept.append((x, y, w, h))
        return [(y, x + w, y + h, x) for (x, y, w, h) in kept]

    def train(self, verbose=True):
        """Entrena cargando imagenes y extrayendo plantillas LBP."""
        self.face_templates = []
        face_names_loaded = []
        stats = {'total_people': 0, 'total_faces': 0, 'total_encodings': 0, 'errors': []}

        if not os.path.exists(KNOWN_FACES_DIR):
            if verbose:
                print(f"Error: directorio {KNOWN_FACES_DIR} no existe")
            return stats

        person_folders = sorted([
            d for d in os.listdir(KNOWN_FACES_DIR)
            if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))
        ])
        if not person_folders:
            if verbose:
                print(f"No hay carpetas de personas en {KNOWN_FACES_DIR}")
            return stats

        stats['total_people'] = len(person_folders)

        for person_name in person_folders:
            person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
            image_paths = get_image_files(person_dir)
            person_count = 0
            seen_hashes = set()  # evitar duplicados exactos

            for image_path in image_paths:
                try:
                    if not is_valid_image(image_path):
                        continue
                    image = load_image(image_path)
                    if image is None:
                        continue

                    # Hash simple para detectar imágenes duplicadas
                    thumb = cv2.resize(
                        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image,
                        (16, 16)
                    )
                    img_hash = hash(thumb.tobytes())
                    if img_hash in seen_hashes:
                        if verbose:
                            print(f"  ~ {person_name}: duplicado ignorado: {os.path.basename(image_path)}")
                        continue
                    seen_hashes.add(img_hash)

                    face_locations = self._detect_faces(image)
                    if not face_locations:
                        if verbose:
                            print(f"  ~ {person_name}: sin cara en {os.path.basename(image_path)}")
                        continue

                    # Tomar solo la cara mas grande de cada imagen
                    face_location = max(
                        face_locations,
                        key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3])
                    )
                    t, r, b, l_ = face_location
                    face_region = image[t:b, l_:r]
                    vec = _preprocess_face(face_region)
                    if vec is not None:
                        self.face_templates.append((person_name, vec))
                        face_names_loaded.append(person_name)
                        person_count += 1
                        stats['total_encodings'] += 1

                    stats['total_faces'] += 1

                except Exception as e:
                    stats['errors'].append(str(e))

            if verbose and person_count > 0:
                print(f"  OK {person_name}: {person_count} plantilla(s) unicas")

        if not self.face_templates:
            if verbose:
                print("Error: no se encontraron caras en las imagenes")
            return stats

        # Calcular umbral adaptativo segun distribucion de similitudes
        self.threshold = self._calibrate_threshold()

        self.known_face_names = list(set(face_names_loaded))
        self.known_face_descriptors = [True] * len(self.face_templates)
        self.save_encodings()

        if verbose:
            print()
            print("=" * 50)
            print(f"Entrenamiento completado:")
            print(f"  Personas: {stats['total_people']}")
            print(f"  Plantillas unicas: {stats['total_encodings']}")
            print(f"  Umbral adaptativo: {self.threshold:.3f}")
            print("=" * 50)

        return stats

    def _calibrate_threshold(self):
        """
        Calcula un umbral de similitud basado en los datos de entrenamiento.
        Busca un valor entre la maxima similitud inter-clase y la minima intra-clase.
        """
        if len(self.face_templates) < 2:
            return DEFAULT_THRESHOLD

        names = [t[0] for t in self.face_templates]
        vecs = np.array([t[1] for t in self.face_templates])
        unique_names = list(set(names))

        if len(unique_names) < 2:
            return DEFAULT_THRESHOLD

        intra_sims = []
        inter_sims = []

        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = float(np.dot(vecs[i], vecs[j]))
                if names[i] == names[j]:
                    intra_sims.append(sim)
                else:
                    inter_sims.append(sim)

        if not inter_sims:
            return DEFAULT_THRESHOLD

        max_inter = max(inter_sims)
        min_intra = min(intra_sims) if intra_sims else max_inter + 0.1

        # Umbral = punto medio entre max inter-clase y min intra-clase
        threshold = (max_inter + min_intra) / 2.0

        # Limitar a rango razonable
        threshold = max(0.60, min(0.92, threshold))
        return threshold

    def _predict_face(self, vec):
        """
        1-NN: encuentra la plantilla mas similar y decide si supera el umbral.
        Retorna (nombre, confianza).
        """
        if not self.face_templates:
            return "Desconocido", 0.0

        vecs = np.array([t[1] for t in self.face_templates])
        names = [t[0] for t in self.face_templates]

        similarities = vecs @ vec  # coseno (vectores ya normalizados)
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self.threshold:
            # Votar entre todas las plantillas de esa persona
            person = names[best_idx]
            person_sims = [s for n, s in zip(names, similarities) if n == person]
            confidence = float(np.mean(person_sims))
            return person, confidence

        return "Desconocido", 0.0

    def recognize_faces(self, image):
        """Reconoce caras en una imagen RGB. Retorna dict con resultados."""
        if not self.known_face_descriptors:
            return {
                "error": "No hay modelo entrenado. Ejecuta: python train.py",
                "faces_detected": 0,
                "faces": []
            }

        results = {'faces_detected': 0, 'faces': []}

        try:
            face_locations = self._detect_faces(image)
            results['faces_detected'] = len(face_locations)

            for face_location in face_locations:
                t, r, b, l_ = face_location
                face_region = image[t:b, l_:r]
                vec = _preprocess_face(face_region)
                if vec is None:
                    continue

                name, confidence = self._predict_face(vec)
                results['faces'].append({
                    'name': name,
                    'confidence': confidence,
                    'similarity': confidence,
                    'location': {'top': t, 'right': r, 'bottom': b, 'left': l_}
                })

        except Exception as e:
            results['error'] = str(e)

        return results

    def save_encodings(self):
        """Guarda el modelo."""
        try:
            model_data = {
                'face_templates': self.face_templates,
                'threshold': self.threshold,
                'known_face_names': self.known_face_names,
            }
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(model_data, f)
            with open(ENCODINGS_FILE, 'wb') as f:
                pickle.dump(self.known_face_descriptors, f)
            with open(NAMES_FILE, 'wb') as f:
                pickle.dump(self.known_face_names, f)
            print("Modelo guardado OK")
        except Exception as e:
            print(f"Error al guardar: {e}")

    def load_encodings(self):
        """Carga el modelo."""
        try:
            if os.path.exists(MODEL_FILE):
                with open(MODEL_FILE, 'rb') as f:
                    data = pickle.load(f)
                # Detectar si es modelo antiguo (PCA+SVM) o nuevo (LBP)
                if 'face_templates' in data:
                    self.face_templates = data['face_templates']
                    self.threshold = data.get('threshold', DEFAULT_THRESHOLD)
                    self.known_face_names = data.get('known_face_names', [])
                    self.known_face_descriptors = [True] * len(self.face_templates)
                    print(f"Modelo cargado: {len(self.face_templates)} plantilla(s), "
                          f"{len(self.known_face_names)} persona(s), umbral={self.threshold:.3f}")
                else:
                    print("Modelo antiguo detectado. Re-entrena con: python train.py")
                    self.known_face_descriptors = []
        except Exception as e:
            print(f"Sin modelo guardado ({e})")

    def get_stats(self):
        return {
            'total_descriptors': len(self.known_face_descriptors),
            'total_people': len(set(self.known_face_names)),
            'known_people': list(set(self.known_face_names)),
            'tolerance': self.tolerance,
            'model': 'LBP + 1-NN'
        }
