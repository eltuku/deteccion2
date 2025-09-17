import sys, os
from pathlib import Path
# --- Ensure 'src' is importable no matter where you run main.py ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
CADIX - Sistema Industrial de Detección de Cadenas v2.0
Punto de entrada principal de la aplicación.

Características principales:
- Detección de eslabones con IA usando YOLO
- Sistema de medición preciso con calibración
- Base de datos SQLite para almacenamiento
- Interfaz profesional con métricas en tiempo real
- Sistema de alertas y notificaciones
- Generación de reportes automáticos
- Backup y recuperación de datos
"""

import sys
import os
import traceback
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.logger import setup_logging
from src.ui.main_window import CADIXMainWindow

def check_dependencies():
    """Verifica que todas las dependencias estén instaladas"""
    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('onnxruntime', 'onnxruntime'),
        ('customtkinter', 'customtkinter'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'matplotlib')
    ]
    
    missing = []
    
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print("ERROR: Faltan las siguientes dependencias:")
        for package in missing:
            print(f"  - {package}")
        print("\nInstala las dependencias con:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def setup_directories():
    """Crea los directorios necesarios"""
    base_dir = Path.home() / "CADIX"
    dirs_to_create = [
        base_dir / "data",
        base_dir / "logs", 
        base_dir / "reports",
        base_dir / "models",
        base_dir / "backups",
        base_dir / "config"
    ]
    
    for directory in dirs_to_create:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

def main():
    """Función principal"""
    try:
        print("=" * 60)
        print("CADIX - Sistema Industrial de Detección de Cadenas v2.0")
        print("=" * 60)
        
        # Verificar dependencias
        print("Verificando dependencias...")
        if not check_dependencies():
            input("\nPresiona Enter para salir...")
            return 1
        
        # Crear directorios
        print("Configurando directorios...")
        setup_directories()
        
        # Configurar logging
        print("Inicializando sistema de logging...")
        logger = setup_logging()
        
        # Iniciar aplicación
        print("Iniciando interfaz gráfica...")
        app = CADIXMainWindow()
        
        logger.info("Aplicación iniciada exitosamente")
        print("✓ Sistema iniciado correctamente")
        print("\nPresiona Ctrl+C para salir en cualquier momento")
        
        # Ejecutar aplicación
        app.run()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nCerrando aplicación...")
        return 0
        
    except Exception as e:
        print(f"\nERROR CRÍTICO: {e}")
        print("\nDetalles del error:")
        print(traceback.format_exc())
        
        try:
            logger = setup_logging()
            logger.critical(f"Error crítico en aplicación: {e}")
            logger.critical(traceback.format_exc())
        except:
            pass
        
        input("\nPresiona Enter para salir...")
        return 1

if __name__ == "__main__":
    sys.exit(main())