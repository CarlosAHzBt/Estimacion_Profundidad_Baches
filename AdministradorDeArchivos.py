import os
import cv2 as cv

class AdministradorArchivos:
    def __init__(self, carpeta_base="Extraccion"):
        self.carpeta_base = carpeta_base
        self.ruta_archivos_bag=None


    def generar_lista_de_bags_bags(self, ruta_archivos_bag):
        #EstaFuncion generara la lista de los archivos bags de donde se extraen las imagenes
        self.ruta_archivos_bag = ruta_archivos_bag
        # Asumiendo que los archivos bags están directamente bajo carpeta_base
        archivos_bags = [os.path.join(self.ruta_archivos_bag, f) for f in os.listdir(self.ruta_archivos_bag)
                            if os.path.isdir(os.path.join(self.ruta_archivos_bag, f))] 
        return archivos_bags
    def obtener_bag_de_origen(self, ruta_image):
        # Esta función obtiene la ruta del bag de origen de la imagen
        # La ruta del bag de origen es "bag" más el nombre de la carpeta que contiene la imagen
        # Esto implica retroceder dos niveles en la jerarquía de directorios desde la ruta de la imagen
        # y luego agregar "bag"

        # Retrocedemos dos niveles en la jerarquía de directorios desde la ruta de la imagen
        ruta_padre = os.path.dirname(os.path.dirname(ruta_image))

        # Obtenemos el nombre de la carpeta que contiene la imagen
        nombre_carpeta = os.path.basename(ruta_padre)

        # Concatenamos "bag" con el nombre de la carpeta
        bag_de_origen = os.path.join("bag", nombre_carpeta +".bag")

        return bag_de_origen
        
    def generar_lista_de_archivosBags(self):
        # Asumiendo que los archivos bags están directamente bajo carpeta_base
        archivos_bags = [os.path.join(self.carpeta_base, f) for f in os.listdir(self.carpeta_base)
                         if os.path.isdir(os.path.join(self.carpeta_base, f))]
        return archivos_bags

    def generar_lista_de_subcarpetas(self, ruta_carpeta_bag):
        # Genera y devuelve una lista de subcarpetas para una carpeta bag dada
        subcarpetas = [os.path.join(ruta_carpeta_bag, f) for f in os.listdir(ruta_carpeta_bag)
                       if os.path.isdir(os.path.join(ruta_carpeta_bag, f))]
        return subcarpetas

    def generar_lista_de_imagenes(self, ruta_carpeta_bag):
        # Devuelve una lista de rutas de imágenes dentro de la carpeta "Imagen" de la carpeta bag
        ruta_carpeta_imagenes = os.path.join(ruta_carpeta_bag, "Imagenes")
        imagenes = [os.path.join(ruta_carpeta_imagenes, f) for f in os.listdir(ruta_carpeta_imagenes)
                    if os.path.isfile(os.path.join(ruta_carpeta_imagenes, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return imagenes
    
    def generar_lista_de_nubes_puntos(self, ruta_subcarpeta):
        # Genera y devuelve una lista de rutas de nubes de puntos dentro de una subcarpeta
        ruta_carpeta_ply = os.path.join(ruta_subcarpeta, "PLY")
        nubes_puntos = [os.path.join(ruta_carpeta_ply, f) for f in os.listdir(ruta_carpeta_ply)
                        if os.path.isfile(os.path.join(ruta_carpeta_ply, f))]
        return nubes_puntos
    
    def imagenes_contorno_bache(self, ruta,image):
        #Guardar la imagen con el contorno del bache
        cv.imwrite(ruta,image)

    def crear_carpeta(self, ruta_carpeta):
        # Crea una carpeta si no existe
        if not os.path.exists(ruta_carpeta):
            os.makedirs(ruta_carpeta)

    def consultar_carpeta(self, ruta_carpeta, tipo='imagenes'):
        # Devuelve la lista de archivos dentro de una carpeta específica, filtrando por tipo
        archivos = [os.path.join(ruta_carpeta, f) for f in os.listdir(ruta_carpeta)
                    if os.path.isfile(os.path.join(ruta_carpeta, f)) and f.endswith(self._obtener_extension(tipo))]
        return archivos

    def _obtener_extension(self, tipo):
        # Devuelve la extensión de archivo basada en el tipo especificado
        if tipo == 'imagenes':
            return ('.png', '.jpg', '.jpeg')
        elif tipo == 'nube_puntos':
            return '.ply'
        else:
            return ''

    def borrar_archivos_extraidos(self):
        # Elimina la carpeta base y su contenido
        if os.path.exists(self.carpeta_base):
            for root, dirs, files in os.walk(self.carpeta_base, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.carpeta_base)
        else:
            print(f"La carpeta {self.carpeta_base} no existe.")