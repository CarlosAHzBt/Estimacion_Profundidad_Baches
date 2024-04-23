
import os
import cv2 as cv
import logging
import shutil

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdministradorArchivos:
    def __init__(self, carpeta_base="Extraccion"):
        self.carpeta_base = carpeta_base

    def generar_lista_de_bags_bags(self, ruta_archivos_bag):
        #EstaFuncion generara la lista de los archivos bags de donde se extraen las imagenes
        self.ruta_archivos_bag = ruta_archivos_bag
        # Asumiendo que los archivos bags están directamente bajo carpeta_base
        archivos_bags = [os.path.join(self.ruta_archivos_bag, f) for f in os.listdir(self.ruta_archivos_bag)
                            if os.path.isdir(os.path.join(self.ruta_archivos_bag, f))] 
        return archivos_bags
    
    def obtener_bag_de_origen(self, ruta_imagen,bag_de_origen):
        # Obtiene la ruta del bag de origen de la imagen
        ruta_padre = os.path.dirname(os.path.dirname(ruta_imagen))
        nombre_carpeta = os.path.basename(ruta_padre)
        bag_de_origen = os.path.join(bag_de_origen + "/", nombre_carpeta + ".bag")
        return bag_de_origen
    def generar_lista_de_archivosBags(self):
        # Asumiendo que los archivos bags están directamente bajo carpeta_base
        archivos_bags = [os.path.join(self.carpeta_base, f) for f in os.listdir(self.carpeta_base)
                         if os.path.isdir(os.path.join(self.carpeta_base, f))]
        return archivos_bags
    def generar_lista_de_subcarpetas(self, ruta_carpeta_bag):
        # Genera y devuelve una lista de subcarpetas para una carpeta bag dada
        try:
            subcarpetas = [os.path.join(ruta_carpeta_bag, f) for f in os.listdir(ruta_carpeta_bag)
                           if os.path.isdir(os.path.join(ruta_carpeta_bag, f))]
            return subcarpetas
        except FileNotFoundError:
            logging.error(f"Directorio no encontrado: {ruta_carpeta_bag}")
            return []

    def generar_lista_de_imagenes(self, ruta_carpeta_bag):
        # Devuelve una lista de rutas de imágenes
        ruta_carpeta_imagenes = os.path.join(ruta_carpeta_bag, "Imagenes")
        try:
            imagenes = [os.path.join(ruta_carpeta_imagenes, f) for f in os.listdir(ruta_carpeta_imagenes)
                        if os.path.isfile(os.path.join(ruta_carpeta_imagenes, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            return imagenes
        except FileNotFoundError:
            logging.error(f"Directorio no encontrado: {ruta_carpeta_imagenes}")
            return []

    def imagenes_contorno_bache(self, ruta, image):
        # Guardar la imagen con el contorno del bache
        cv.imwrite(ruta, image)

    def crear_carpeta(self, ruta_carpeta):
        # Crea una carpeta si no existe
        os.makedirs(ruta_carpeta, exist_ok=True)

    def borrar_archivos_extraidos(self):
        # Elimina la carpeta base y su contenido
        shutil.rmtree(self.carpeta_base)