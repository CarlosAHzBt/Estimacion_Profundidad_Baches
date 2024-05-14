import logging
import os
import pyrealsense2 as rs
import numpy as np
import cv2
# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class BagFile:
    def __init__(self, bag_file_path, base_folder, lock):
        self.bag_file_path = bag_file_path
        self.images_folder, self.ply_folder, self.depth_folder = self.create_output_folders(bag_file_path, base_folder)
        self.align = rs.align(rs.stream.color)  # Crear objeto de alineación una sola vez
        self.pc = rs.pointcloud()  # Crear objeto de nube de puntos una sola vez
        self.lock = lock  # Lock para sincronización

    @staticmethod
    def create_output_folders(bag_file_path, base_folder):
        bag_name = os.path.splitext(os.path.basename(bag_file_path))[0]
        images_folder = os.path.join(base_folder, bag_name, 'Imagenes')
        depth_folder = os.path.join(base_folder, bag_name, 'ImagenesProfundidad')
        ply_folder = os.path.join(base_folder, bag_name, 'Ply')
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(depth_folder, exist_ok=True)
        os.makedirs(ply_folder, exist_ok=True)
        return images_folder, ply_folder, depth_folder

    def configure_pipeline(self):
        config = rs.config()
        config.enable_device_from_file(self.bag_file_path, repeat_playback=False)
        pipeline = rs.pipeline()
        pipeline.start(config)
        playback = pipeline.get_active_profile().get_device().as_playback()
        playback.set_real_time(False)
        return pipeline

    def process_bag_file(self):
        pipeline = self.configure_pipeline()
        frame_number = 0
        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame or not color_frame:
                    continue

                # Bloquear el acceso para evitar duplicación de frames
                with self.lock:
                    self.save_color_image(color_frame, frame_number)
                    self.save_depth_frame(depth_frame, frame_number)

                frame_number += 1
                logging.info(f"Frame {frame_number} procesado.")
        except RuntimeError:
            print("Todos los frames han sido procesados.")
        finally:
            pipeline.stop()

    def save_color_image(self, color_frame, frame_number):
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Cambio a RGB para coherencia con OpenCV
        filename = f'{self.images_folder}/frame_{frame_number:05d}.png'
        if not os.path.exists(filename):
            cv2.imwrite(filename, color_image)
        else:
            logging.warning(f"El archivo {filename} ya existe, se está evitando la duplicación.")

    def save_depth_frame(self, depth_frame, frame_number):
        depth_image = np.asanyarray(depth_frame.get_data())
        filename = f'{self.depth_folder}/frame_{frame_number:05d}.png'
        if not os.path.exists(filename):
            cv2.imwrite(filename, depth_image)
        else:
            logging.warning(f"El archivo {filename} ya existe, se está evitando la duplicación.")
