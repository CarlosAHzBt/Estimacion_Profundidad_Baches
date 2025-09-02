"""Point cloud processing and filtering utilities."""

import logging
from typing import Tuple, Optional, List
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class PointCloudProcessor:
    """Processes and filters point clouds for pothole analysis."""

    def __init__(self):
        """Initialize the point cloud processor."""
        pass

    def start_pipeline(self, bag_file_path: str) -> rs.pipeline:
        """Start RealSense pipeline for reading from .bag file.

        Args:
            bag_file_path: Path to the .bag file

        Returns:
            Started pipeline
        """
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_file_path)
        config.enable_stream(rs.stream.depth)
        return pipeline.start(config)

    def get_intrinsics_from_pipeline(self, profile: rs.pipeline_profile) -> Tuple[rs.intrinsics, float]:
        """Get camera intrinsics and depth scale from pipeline.

        Args:
            profile: Active pipeline profile

        Returns:
            Tuple of (intrinsics, depth_scale)
        """
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        intrinsics = depth_profile.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        return intrinsics, depth_scale

    def depth_image_to_pointcloud(self, depth_image: np.ndarray,
                                intrinsics: rs.intrinsics,
                                depth_scale: float) -> o3d.geometry.PointCloud:
        """Convert depth image to point cloud.

        Args:
            depth_image: Depth image array
            intrinsics: Camera intrinsics
            depth_scale: Depth scale factor

        Returns:
            Open3D point cloud
        """
        depth_o3d = o3d.geometry.Image(depth_image)
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height,
            intrinsics.fx, intrinsics.fy,
            intrinsics.ppx, intrinsics.ppy
        )
        return o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, o3d_intrinsics
        )

    def get_bounding_box(self, contour_coords: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Generate bounding box from contour coordinates.

        Args:
            contour_coords: Contour coordinates array

        Returns:
            Bounding box coordinates ((x_min, y_min), (x_max, y_max))
        """
        coords = np.array(contour_coords)
        return (np.min(coords, axis=0), np.max(coords, axis=0))

    def pixel_to_point(self, intrinsics: rs.intrinsics, u: int, v: int, depth: float) -> Optional[List[float]]:
        """Convert pixel coordinates to 3D point.

        Args:
            intrinsics: Camera intrinsics
            u: Pixel u coordinate
            v: Pixel v coordinate
            depth: Depth value

        Returns:
            3D point coordinates or None if depth is invalid
        """
        if depth > 0:
            return rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
        return None

    def crop_pointcloud(self, pcd: o3d.geometry.PointCloud,
                       intrinsics: rs.intrinsics,
                       depth_image: np.ndarray,
                       bounding_box: Tuple[Tuple[float, float], Tuple[float, float]],
                       depth_scale: float,
                       rotation_matrix: np.ndarray,
                       center: Tuple[float, float, float] = (0, 0, 0)) -> o3d.geometry.PointCloud:
        """Crop point cloud using bounding box and apply rotation.

        Args:
            pcd: Input point cloud
            intrinsics: Camera intrinsics
            depth_image: Depth image array
            bounding_box: Bounding box coordinates
            depth_scale: Depth scale factor
            rotation_matrix: Rotation matrix to apply
            center: Rotation center

        Returns:
            Cropped and rotated point cloud
        """
        (box1, box2) = bounding_box
        (x_min, y_min) = box1
        (x_max, y_max) = box2

        spatial_points = []
        for u in range(int(x_min), int(x_max)):
            for v in range(int(y_min), int(y_max)):
                if depth_image[v, u] > 0:
                    point = self.pixel_to_point(intrinsics, u, v, depth_image[v, u] * depth_scale)
                    if point is not None:
                        spatial_points.append(point)

        pcd_cropped = o3d.geometry.PointCloud()
        pcd_cropped.points = o3d.utility.Vector3dVector(spatial_points)
        pcd_cropped.rotate(rotation_matrix, center=center)

        return pcd_cropped


class RANSACProcessor:
    """RANSAC-based plane segmentation and leveling."""

    def __init__(self, distance_threshold: float = 0.05):
        """Initialize RANSAC processor.

        Args:
            distance_threshold: Distance threshold for RANSAC
        """
        self.distance_threshold = distance_threshold

    def segment_plane(self, pcd: o3d.geometry.PointCloud,
                     distance_thresh: float = 0.01,
                     ransac_n: int = 100,
                     num_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Segment plane using RANSAC.

        Args:
            pcd: Input point cloud
            distance_thresh: Distance threshold
            ransac_n: Number of points for RANSAC
            num_iterations: Number of iterations

        Returns:
            Tuple of (plane_model, inliers)
        """
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_thresh,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        return plane_model, inliers

    def calculate_rotation_vector(self, plane_normal: np.ndarray,
                                target_vector: List[float] = [0, 0, 1]) -> Tuple[np.ndarray, float]:
        """Calculate rotation axis and angle to align plane normal with target.

        Args:
            plane_normal: Normal vector of the plane
            target_vector: Target vector to align with

        Returns:
            Tuple of (axis, angle)
        """
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        dot_product = np.dot(plane_normal, target_vector)
        axis = np.cross(plane_normal, target_vector)
        angle = np.arccos(dot_product / (np.linalg.norm(plane_normal) * np.linalg.norm(target_vector)))
        return axis, angle

    def apply_rotation(self, pcd: o3d.geometry.PointCloud,
                      axis: np.ndarray,
                      angle: float,
                      center: List[float] = [0, 0, 0]) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Apply rotation to point cloud.

        Args:
            pcd: Input point cloud
            axis: Rotation axis
            angle: Rotation angle
            center: Rotation center

        Returns:
            Tuple of (rotated_point_cloud, rotation_matrix)
        """
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        pcd.rotate(rotation_matrix, center=center)
        return pcd, rotation_matrix

    def segment_and_level(self, pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Segment plane and level the point cloud.

        Args:
            pcd: Input point cloud

        Returns:
            Tuple of (leveled_point_cloud, rotation_matrix)
        """
        plane_model, _ = self.segment_plane(pcd)
        plane_normal = np.array(plane_model[:3])
        axis, angle = self.calculate_rotation_vector(plane_normal)
        leveled_pcd, rotation_matrix = self.apply_rotation(pcd, axis, angle)
        return leveled_pcd, rotation_matrix
