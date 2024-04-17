from shapely.geometry import Point, Polygon
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2

class PointCloudFilter:
    def __init__(self):
        """
        Constructor de la clase PointCloudFilter.
        """
        pass

    def start_pipeline(self, bag_file_path):
        """
        Inicia la pipeline de RealSense para leer datos desde un archivo .bag.

        Parámetros:
        - bag_file_path (str): Ruta al archivo .bag.
        
        Retorna:
        - rs.pipeline: Pipeline configurada y en ejecución.
        """
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_file_path)
        config.enable_stream(rs.stream.depth)
        return pipeline.start(config)

    def obtener_intrinsecos_from_pipeline(self, profile):
        """
        Obtiene los intrínsecos de la cámara desde una pipeline activa.

        Parámetros:
        - pipeline (rs.pipeline): Pipeline de RealSense en ejecución.

        Retorna:
        - tuple: Intrínsecos de la cámara y escala de profundidad.
        """
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        intrinsics = depth_profile.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        return intrinsics, depth_scale

    def depth_image_to_pointcloud(self, depth_image, intrinsics, depth_scale):
        """
        Convierte una imagen de profundidad en una nube de puntos.

        Parámetros: 
        - depth_image (rs.depth_frame): Imagen de profundidad.
        - intrinsics (rs.intrinsics): Intrínsecos de la cámara.
        - depth_scale (float): Escala de profundidad.

        Retorna:
        - np.array: Nube de puntos.
        """
        depth_o3d = o3d.geometry.Image(depth_image)
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        return o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, o3d_intrinsics)

    def get_bounding_box(self, contour_coords_meters):
        """
        Genera un bounding box a partir de las coordenadas de un contorno.

        Parámetros:
        - contour_coords_meters (list): Coordenadas del contorno en metros.

        Retorna: 
        - tuple: Coordenadas del bounding box.
        """
        coords = np.array(contour_coords_meters)
        return np.min(coords, axis=0), np.max(coords, axis=0)

    def pixel_to_point(self, intrinsics, u, v, depth):
        """
        Convierte un píxel en coordenadas de profundidad a un punto espacial 3D.

        Parámetros:
        - intrinsics (rs.intrinsics): Intrínsecos de la cámara.
        - u (int): Coordenada u del píxel.
        - v (int): Coordenada v del píxel.
        - depth (float): Valor de profundidad del píxel.

        Retorna:
        - list: Punto 3D en el espacio.
        """
        if depth > 0:
            return rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
        return None

    def recortar_nube_de_puntos(self, pcd, intrinsics, depth_image, bounding_box, depth_scale, R, centro=(0,0,0)):
        """
        Recorta la nube de puntos a partir de un bounding box en una imagen de profundidad y rota la nube recortada otra vez..

        Parámetros:
        - pcd (o3d.geometry.PointCloud): Nube de puntos.
        - intrinsics (rs.intrinsics): Intrínsecos de la cámara.
        - depth_image (np.array): Imagen de profundidad.
        - bounding_box (tuple): Coordenadas del bounding box.
        - depth_scale (float): Escala de profundidad.
        - R (np.array): Matriz de rotación.
        - centro (tuple): Centro de rotación.

        """
        (box1, box2) = bounding_box
        (x_min, y_min) = box1
        (x_max, y_max) = box2       
        puntos_espaciales = [self.pixel_to_point(intrinsics, u, v, depth_image[v, u] * depth_scale)
                             for u in range(x_min, x_max) for v in range(y_min, y_max) if depth_image[v, u] > 0]

        pcd_cropped = o3d.geometry.PointCloud()
        pcd_cropped.points = o3d.utility.Vector3dVector([p for p in puntos_espaciales if p is not None])
        #o3d.visualization.draw_geometries([pcd_cropped])
        pcd_cropped.rotate(R, center=centro)
        #o3d.visualization.draw_geometries([pcd_cropped])
        return pcd_cropped
