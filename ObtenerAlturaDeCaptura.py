#Clase para obtener altura de captura sacando la media de puntos de la nube de puntos PLY
import open3d as o3d
import numpy as np
 

class AlturaCaptura:
    def __init__(self):
        pass
    def cargar_nube_puntos(self,archivo_ply):
        # Cargar archivo PLY y retornar nube de puntos
        nube_puntos = o3d.io.read_point_cloud(archivo_ply)
        return nube_puntos
    
    def calcular_altura(self,archivo_ply):
        #Si self.archivo_ply es un string, cargar la nube de puntos
        if isinstance(archivo_ply, str):
            nube_puntos = self.cargar_nube_puntos()
        else:
            nube_puntos= archivo_ply
        
        puntos = np.asarray(nube_puntos.points)
        
        # Calcular la moda de las alturas en el eje Z
        puntos = np.asarray(nube_puntos.points)
        superficie_estimada = np.median(puntos[:, 2])
        return superficie_estimada
    def calcular_altura_apartir_de_depth_image(self,depth_image):
        # Calcular la moda de las alturas en el eje Z
        puntos = np.asarray(depth_image)
        superficie_estimada = np.median(puntos)
        return superficie_estimada/-1000



 