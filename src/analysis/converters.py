"""Conversion utilities for pixel to metric transformations."""

import math
from typing import Tuple, Union
import numpy as np
import open3d as o3d


class PixelToMeterConverter:
    """Converts pixel measurements to metric units."""

    def __init__(self,
                 fov_horizontal: float = 69.0,
                 fov_vertical: float = 42.0,
                 resolution_width: int = 848,
                 resolution_height: int = 480):
        """Initialize the converter.

        Args:
            fov_horizontal: Horizontal field of view in degrees
            fov_vertical: Vertical field of view in degrees
            resolution_width: Image width in pixels
            resolution_height: Image height in pixels
        """
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height

    def calculate_scale(self, capture_height: float) -> Tuple[float, float]:
        """Calculate horizontal and vertical scale factors.

        Args:
            capture_height: Height of camera capture in meters

        Returns:
            Tuple of (horizontal_scale, vertical_scale) in meters per pixel
        """
        real_width = 2 * capture_height * math.tan(math.radians(self.fov_horizontal / 2))
        real_height = 2 * capture_height * math.tan(math.radians(self.fov_vertical / 2))

        horizontal_scale = real_width / self.resolution_width
        vertical_scale = real_height / self.resolution_height

        return horizontal_scale, vertical_scale

    def convert_radius_pixels_to_meters(self, radius_pixels: float, horizontal_scale: float) -> float:
        """Convert pixel radius to meters.

        Args:
            radius_pixels: Radius in pixels
            horizontal_scale: Horizontal scale factor

        Returns:
            Radius in meters
        """
        return radius_pixels * horizontal_scale


class CaptureHeightEstimator:
    """Estimates camera capture height from point clouds or depth images."""

    def __init__(self):
        """Initialize the height estimator."""
        pass

    def estimate_from_ply(self, ply_path: Union[str, o3d.geometry.PointCloud]) -> float:
        """Estimate capture height from PLY file or point cloud.

        Args:
            ply_path: Path to PLY file or point cloud object

        Returns:
            Estimated capture height in meters
        """
        if isinstance(ply_path, str):
            point_cloud = o3d.io.read_point_cloud(ply_path)
        else:
            point_cloud = ply_path

        points = np.asarray(point_cloud.points)
        estimated_surface = np.median(points[:, 2])
        return estimated_surface

    def estimate_from_depth_image(self, depth_image: np.ndarray) -> float:
        """Estimate capture height from depth image.

        Args:
            depth_image: Depth image array

        Returns:
            Estimated capture height in meters
        """
        points = np.asarray(depth_image)
        estimated_surface = np.median(points)
        return estimated_surface / -1000
