import os
import cv2 as cv
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdministradorArchivos:
    def __init__(self, carpeta_base="Extraccion", carpeta_salida="Resultados"):
        self.carpeta_base = carpeta_base
        self.carpeta_salida = carpeta_salida

    def generar_lista_de_bags(self, ruta_archivos_bag=None):
        ruta_busqueda = ruta_archivos_bag if ruta_archivos_bag else self.carpeta_base
        archivos_bags = [os.path.join(ruta_busqueda, f) for f in os.listdir(ruta_busqueda)
                         if os.path.isfile(os.path.join(ruta_busqueda, f)) and f.endswith('.bag')]
        return archivos_bags

    def obtener_bag_de_origen(self, ruta_imagen):
        ruta_padre = os.path.dirname(os.path.dirname(ruta_imagen))
        nombre_carpeta = os.path.basename(ruta_padre)
        bag_de_origen = os.path.join("bag", nombre_carpeta + ".bag")
        return bag_de_origen

    def generar_lista_de_imagenes(self, ruta_carpeta_bag):
        ruta_carpeta_imagenes = os.path.join(ruta_carpeta_bag, "Imagenes")
        try:
            imagenes = [os.path.join(ruta_carpeta_imagenes, f) for f in os.listdir(ruta_carpeta_imagenes)
                        if os.path.isfile(os.path.join(ruta_carpeta_imagenes, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            return imagenes
        except FileNotFoundError:
            logging.error(f"Directorio no encontrado: {ruta_carpeta_imagenes}")
            return []

    def imagenes_contorno_bache(self, ruta, image):
        cv.imwrite(ruta, image)

    def crear_carpeta(self, ruta_carpeta):
        os.makedirs(ruta_carpeta, exist_ok=True)

    def borrar_archivos_extraidos(self):
        try:
            os.rmdir(self.carpeta_base)
        except OSError as e:
            logging.error(f"Error al eliminar {self.carpeta_base}: {e}")
