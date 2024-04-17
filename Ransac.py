import numpy as np
import open3d as o3d

class RANSAC:
    """
    Clase RANSAC que realiza la detección y nivelación del plano del terreno en una nube de puntos preprocesada.
    Attributes:
        distancia_thresh (float): Umbral de distancia para el algoritmo RANSAC.
    """

    def __init__(self, distancia_thresh=0.05):
        """
        Inicializa la clase RANSAC con el umbral de distancia para el algoritmo RANSAC.
        Parameters:
            distancia_thresh (float): Umbral de distancia para el algoritmo RANSAC.
        """
        self.distancia_thresh = distancia_thresh

    #def filtrar_puntos(self, pcd, z_min, z_max):
    #    """
    #    Filtra la nube de puntos en el eje Z entre z_min y z_max.
    #    """
    #    puntos = np.asarray(pcd.points)
    #    filtrados = puntos[(puntos[:, 1] > z_min) & (puntos[:, 1] < z_max)]
    #    pcd_filtrado = o3d.geometry.PointCloud()
    #    pcd_filtrado.points = o3d.utility.Vector3dVector(filtrados)
    #    return pcd_filtrado
#
    #def segmentar_terreno(self, pcd):
    #    """
    #    Segmenta el terreno utilizando el algoritmo RANSAC.
    #    """
    #    plano, inliers = pcd.segment_plane(distance_threshold=self.distancia_thresh, ransac_n=100, num_iterations=1000)
#
    #    inlier_cloud = pcd.select_by_index(inliers)
#
    #    return inlier_cloud, plano
#
    #def nivelar_puntos(self, plano):
    #    """
    #    Nivela la nube de puntos basándose en el plano del terreno detectado.
    #    """
    #    A, B, C, D = plano
    #    norm = np.linalg.norm([A, B, C])
    #    vector_plano = np.array([A, B, C]) / norm
    #    up_vector = np.array([0, 0, 1])
    #    rot = self.matriz_rotacion(vector_plano, up_vector)
    #    R = rot
    #    transform = np.eye(4)
    #    transform[:3, :3] = rot
    #
    #    return transform,R  # Retorna la nube de puntos transformada ya que la transformación se aplica in situ
#
    #@staticmethod
    #def matriz_rotacion(v1, v2):
    #    """
    #    Calcula la matriz de rotación para alinear v1 con v2.
    #    """
    #    v = np.cross(v1, v2)
    #    s = np.linalg.norm(v)
    #    c = np.dot(v1, v2)
    #    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    #    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    #    return R
#
    #def procesar_nube_completa(self, pcd, z_min=-2, z_max=2):
    #    """
    #    Procesa la nube de puntos completa: filtra, segmenta y nivela el terreno.
    #    """
    #    # Filtrar puntos en el eje Z
    #    pcd_filtrado = self.filtrar_puntos(pcd, z_min, z_max)
#
#
    #    # Segmentar terreno
    #    pcd_terreno, plano = self.segmentar_terreno(pcd_filtrado)
#
    #    # Nivelar puntos
    #    transformacion,R = self.nivelar_puntos(plano)
    #    pcd_nivelada = pcd.transform(transformacion) # La nube de puntos ya está nivelada después de esta llamada
    #    #o3d.visualization.draw_geometries([pcd_nivelada])
#
    #    return pcd_nivelada, R
    #
    def segmentar_plano(self,pcd, distancia_thresh=0.01, ransac_n=100, num_iterations=1000):
        """ Segmenta el plano usando RANSAC y devuelve el modelo del plano y los inliers. """
        plano_modelo, inliers = pcd.segment_plane(distance_threshold=distancia_thresh, ransac_n=ransac_n, num_iterations=num_iterations)
        return plano_modelo, inliers

    def calcular_vector_rotacion(self,normal_del_plano, vector_objetivo=[0, 0, 1]):
        """ Calcula el ángulo y el eje para la rotación que alinea la normal del plano con el vector objetivo. """
        normal_del_plano = normal_del_plano / np.linalg.norm(normal_del_plano)  # Normaliza la normal del plano
        producto_punto = np.dot(normal_del_plano, vector_objetivo)
        axis = np.cross(normal_del_plano, vector_objetivo)
        angle = 1 + np.arccos(producto_punto / (np.linalg.norm(normal_del_plano) * np.linalg.norm(vector_objetivo)))
        return axis, angle

    def aplicar_rotacion(self, pcd, axis, angle, centro=[0, 0, 0]):
        """ Aplica la rotación a la nube de puntos y devuelve la nube de puntos rotada. """
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        pcd.rotate(R, center=centro)
        return pcd, R

    def segmentar_plano_y_nivelar(self,pcd):
        """ Combina los pasos para segmentar un plano y nivelar la nube de puntos basada en ese plano. """
        plano_modelo, _ = self.segmentar_plano(pcd)
        normal_del_plano = np.array(plano_modelo[:3])  # Extrae la normal del modelo del plano
        axis, angle = self.calcular_vector_rotacion(normal_del_plano)
        pcd_rotada, R = self.aplicar_rotacion(pcd, axis, angle)
        return pcd_rotada, R

    def segmentar_plano_y_nivelar(self,pcd):
        plano_modelo, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=100, num_iterations=1000)
        [a, b, c, d] = plano_modelo
        normal_del_plano = np.array([a, b, c])
        normal_del_plano /= np.linalg.norm(normal_del_plano)
        vector_objetivo = [0, 0, 1]
        producto_punto = np.dot(normal_del_plano, vector_objetivo)
        axis = np.cross(normal_del_plano, vector_objetivo)
        angle =1 + np.arccos(producto_punto / (np.linalg.norm(normal_del_plano) * np.linalg.norm(vector_objetivo)))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        centro = [0,0,0] # O ajustar según sea necesario
        pcd.rotate(R, center=centro)
        return pcd, R