"""
Descargador automático de Haar Cascades
Resuelve problema de OpenCV en Windows Python 3.14
"""

import os
import urllib.request
from pathlib import Path


def download_haar_cascades():
    """Descarga los Haar Cascades necesarios."""
    
    cascades_dir = Path("data") / "cascades"
    cascades_dir.mkdir(parents=True, exist_ok=True)
    
    cascade_file = cascades_dir / "haarcascade_frontalface_default.xml"
    
    if cascade_file.exists():
        print(f"✓ Haar Cascade ya existe: {cascade_file}")
        return cascade_file
    
    print("📥 Descargando Haar Cascade...")
    
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    
    try:
        urllib.request.urlretrieve(url, cascade_file)
        print(f"✓ Descargado: {cascade_file}")
        return cascade_file
    except Exception as e:
        print(f"✗ Error al descargar: {e}")
        return None


if __name__ == "__main__":
    download_haar_cascades()
