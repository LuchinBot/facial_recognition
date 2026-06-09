"""
Configuración global de la aplicación
"""
import os

# Rutas base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
KNOWN_FACES_DIR = os.path.join(DATA_DIR, 'known_faces')
ENCODINGS_DIR = os.path.join(DATA_DIR, 'face_encodings')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Crear carpetas si no existen
for directory in [KNOWN_FACES_DIR, ENCODINGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Parámetros de reconocimiento
FACE_RECOGNITION_TOLERANCE = 0.6  # Tolerancia para coincidencias (0-1, menor = más estricto)
MODEL = 'hog'  # 'hog' es rápido, 'cnn' es más preciso pero más lento

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

# Configuración de modelos de encoding
ENCODINGS_FILE = os.path.join(ENCODINGS_DIR, 'encodings.pkl')
NAMES_FILE = os.path.join(ENCODINGS_DIR, 'names.pkl')
