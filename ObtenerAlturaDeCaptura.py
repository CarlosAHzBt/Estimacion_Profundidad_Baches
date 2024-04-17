#Clase para obtener altura de captura sacando la media de puntos de la nube de puntos PLY
import open3d as o3d
import numpy as np
 

class AlturaCaptura:
    def __init__(self, archivo_ply):
        self.archivo_ply = archivo_ply

    def cargar_nube_puntos(self):
        # Cargar archivo PLY y retornar nube de puntos
        nube_puntos = o3d.io.read_point_cloud(self.archivo_ply)
        return nube_puntos
    
    def calcular_altura(self):
        #Si self.archivo_ply es un string, cargar la nube de puntos
        if isinstance(self.archivo_ply, str):
            nube_puntos = self.cargar_nube_puntos()
        else:
            nube_puntos= self.archivo_ply
        
        puntos = np.asarray(nube_puntos.points)
        
        # Calcular la moda de las alturas en el eje Z
        puntos = np.asarray(nube_puntos.points)
        superficie_estimada = np.median(puntos[:, 2])
        return superficie_estimada



 