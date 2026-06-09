"""
Funciones auxiliares para reconocimiento facial
"""
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from src.config import ALLOWED_EXTENSIONS


def get_image_files(directory):
    """
    Obtiene todos los archivos de imagen de un directorio.
    
    Args:
        directory: Ruta del directorio
        
    Returns:
        Lista de rutas de archivos de imagen
    """
    image_files = []
    
    if not os.path.exists(directory):
        return image_files
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                image_files.append(file_path)
    
    return image_files


def load_image(image_path):
    """
    Carga una imagen desde un archivo.
    
    Args:
        image_path: Ruta de la imagen
        
    Returns:
        Array de la imagen en RGB o None si falla
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error al cargar imagen {image_path}: {e}")
        return None


def save_image(image_path, image):
    """
    Guarda una imagen en un archivo.
    
    Args:
        image_path: Ruta donde guardar
        image: Array de la imagen en RGB
    """
    try:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image_bgr)
    except Exception as e:
        print(f"Error al guardar imagen {image_path}: {e}")


def get_person_name_from_path(image_path):
    """
    Extrae el nombre de la persona del path de la imagen.
    Espera estructura: known_faces/nombre_persona/foto.jpg
    
    Args:
        image_path: Ruta de la imagen
        
    Returns:
        Nombre de la persona
    """
    # Obtener la carpeta padre (nombre de la persona)
    person_name = os.path.basename(os.path.dirname(image_path))
    return person_name


def is_valid_image(image_path):
    """
    Valida que sea una imagen válida.
    
    Args:
        image_path: Ruta de la imagen
        
    Returns:
        True si es válida, False en caso contrario
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False
