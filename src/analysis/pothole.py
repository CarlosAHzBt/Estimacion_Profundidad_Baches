"""Pothole detection and analysis."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import open3d as o3d
from .converters import PixelToMeterConverter, CaptureHeightEstimator
from .filters import PointCloudProcessor, RANSACProcessor
from ..utils.file_manager import FileManager

logger = logging.getLogger(__name__)


class Pothole:
    """Represents a detected pothole with analysis capabilities."""

    def __init__(self,
                 bag_origin: str,
                 bag_path: str,
                 output_folder: str,
                 rgb_image_path: str,
                 pothole_id: str,
                 coordinates: Optional[np.ndarray] = None,
                 min_diameter_mm: float = 80.0):
        """Initialize a pothole instance.

        Args:
            bag_origin: Name of the bag file (without .bag extension)
            bag_path: Path to the .bag file
            output_folder: Output folder for results
            rgb_image_path: Path to RGB image
            pothole_id: Unique identifier for the pothole
            coordinates: Pixel coordinates of the pothole
            min_diameter_mm: Minimum diameter threshold in mm
        """
        self.pothole_id = pothole_id
        self.bag_origin = bag_origin
        self.bag_path = bag_path
        self.extracted_bag_path = Path(output_folder) / "Extraction" / bag_origin  # Path to extracted data
        self.rgb_image_path = rgb_image_path
        self.output_folder = Path(output_folder)
        self.coordinates = np.array(coordinates) if coordinates is not None else np.empty((0, 2), dtype=int)
        self.image_shape = (480, 848)  # Standard resolution

        # Analysis results
        self.contour: Optional[np.ndarray] = None
        self.center_circle: Optional[Tuple[int, int]] = None
        self.circle_radius_pixels: Optional[float] = None
        self.max_radius_mm: Optional[float] = None
        self.estimated_depth: Optional[float] = None
        self.capture_height: Optional[float] = None
        self.horizontal_scale: Optional[float] = None

        # Tracking
        self.grouped_frames: List[str] = [pothole_id]

        # Utilities
        self.converter = PixelToMeterConverter()
        self.height_estimator = CaptureHeightEstimator()
        self.pc_processor = PointCloudProcessor()
        self.ransac = RANSACProcessor()
        self.file_manager = FileManager()

        # Thresholds
        self.min_diameter_mm = min_diameter_mm

    def process_pothole(self) -> bool:
        """Process the pothole and calculate its properties.

        Returns:
            True if processing was successful, False otherwise
        """
        try:
            self._calculate_contour()
            if self.contour is None or not self.contour.size:
                return False

            cropped_pcd = self._crop_and_process_pointcloud()
            if cropped_pcd is not None:
                success = self._estimate_depth(cropped_pcd)
                if success:
                    self._update_grouped_data()
                    return True
            return False
        except Exception as e:
            logger.error(f"Error processing pothole {self.pothole_id}: {e}")
            return False

    def _calculate_contour(self) -> None:
        """Calculate the contour from coordinates."""
        if self.coordinates.size == 0:
            raise ValueError("No coordinates available for contour calculation")

        mask = np.zeros(self.image_shape[:2], dtype=np.uint8)
        for x, y in self.coordinates:
            if 0 <= x < self.image_shape[0] and 0 <= y < self.image_shape[1]:
                mask[x, y] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            external_contour = max(contours, key=cv2.contourArea).squeeze()
            if external_contour.ndim == 1:
                external_contour = external_contour.reshape(-1, 1, 2)
            self.contour = external_contour

    def _crop_and_process_pointcloud(self) -> Optional[o3d.geometry.PointCloud]:
        """Crop and process the point cloud for depth estimation.

        Returns:
            Cropped point cloud or None if processing fails
        """
        # Load depth image
        depth_image = self._load_depth_image()
        if depth_image is None:
            return None

        # Start pipeline and get intrinsics
        pipeline = self.pc_processor.start_pipeline(str(self.bag_path))
        intrinsics, depth_scale = self.pc_processor.get_intrinsics_from_pipeline(pipeline)

        # Convert depth image to point cloud
        pcd = self.pc_processor.depth_image_to_pointcloud(depth_image, intrinsics, depth_scale)

        # Segment plane and level
        leveled_pcd, rotation_matrix = self.ransac.segment_and_level(pcd)

        # Calculate capture height and scale
        points = np.asarray(leveled_pcd.points)
        self.capture_height = np.median(points[:, 2])
        self.horizontal_scale, _ = self.converter.calculate_scale(self.capture_height)

        # Calculate max radius
        self._calculate_max_radius()
        if self.max_radius_mm * 2 < self.min_diameter_mm:
            return None

        # Get bounding box and crop
        bounding_box = self.pc_processor.get_bounding_box(self.contour)
        cropped_pcd = self.pc_processor.crop_pointcloud(
            leveled_pcd, intrinsics, depth_image, bounding_box,
            depth_scale, rotation_matrix
        )

        pipeline.stop()
        return cropped_pcd

    def _load_depth_image(self) -> Optional[np.ndarray]:
        """Load the corresponding depth image.

        Returns:
            Depth image array or None if not found
        """
        # Extract frame number from pothole_id
        frame_name = self.pothole_id.split('_')[0]
        depth_path = self.extracted_bag_path / "ImagenesProfundidad" / f"{frame_name}.png"

        if depth_path.exists():
            return cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        else:
            logger.error(f"Depth image not found: {depth_path}")
            return None

    def _calculate_max_radius(self) -> None:
        """Calculate the maximum radius of the pothole."""
        if self.contour is None:
            raise ValueError("Contour must be calculated before radius calculation")

        contour_image = np.zeros(self.image_shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_image, [self.contour], -1, color=255, thickness=-1)

        points_inside = np.argwhere(contour_image == 255)
        max_radius = 0

        for point in points_inside:
            dist = cv2.pointPolygonTest(self.contour, (int(point[1]), int(point[0])), True)
            if dist > max_radius:
                max_radius = dist
                self.center_circle = (int(point[1]), int(point[0]))

        if max_radius == 0:
            raise ValueError("No points found inside contour for radius calculation")

        self.circle_radius_pixels = max_radius
        self.max_radius_mm = self.converter.convert_radius_pixels_to_meters(
            max_radius, self.horizontal_scale
        ) * 1000  # Convert to mm

        logger.info(f"Pothole {self.pothole_id} diameter: {self.max_radius_mm * 2:.1f} mm")

    def _estimate_depth(self, cropped_pcd: o3d.geometry.PointCloud) -> bool:
        """Estimate the depth of the pothole.

        Args:
            cropped_pcd: Cropped point cloud

        Returns:
            True if estimation was successful
        """
        points = np.asarray(cropped_pcd.points)
        z_values = points[:, 2]
        depth = self.capture_height - np.max(z_values)

        logger.info(f"Pothole {self.pothole_id} depth: {depth:.3f} m "
                   f"(capture height: {self.capture_height:.3f} m)")

        self.estimated_depth = depth
        return True

    def _update_grouped_data(self) -> None:
        """Update grouped data lists (for tracking)."""
        # These would be used when merging with other detections
        pass

    def generate_annotated_image(self) -> Optional[str]:
        """Generate image with contour and circle annotations.

        Returns:
            Path to the annotated image or None if failed
        """
        image = cv2.imread(self.rgb_image_path)
        if image is None:
            logger.error("Could not load RGB image")
            return None

        # Create output directories
        rgb_output_dir = self.output_folder / "Results" / "imagenesRGB" / self.bag_origin
        contour_output_dir = self.output_folder / "Results" / "imagenesContorno" / self.bag_origin

        rgb_output_dir.mkdir(parents=True, exist_ok=True)
        contour_output_dir.mkdir(parents=True, exist_ok=True)

        # Save original RGB image
        rgb_filename = f"{self.pothole_id}_rgb_image.png"
        rgb_path = rgb_output_dir / rgb_filename
        cv2.imwrite(str(rgb_path), image)

        # Draw contour and circle
        cv2.drawContours(image, [self.contour], -1, (0, 255, 0), 2)  # Green contour
        if self.center_circle is not None and self.circle_radius_pixels is not None:
            try:
                cv2.circle(image, self.center_circle, int(self.circle_radius_pixels),
                          (0, 255, 0), 2)  # Green circle
            except TypeError as e:
                logger.error(f"Error drawing circle: {e}")

        # Save annotated image
        contour_filename = f"{self.pothole_id}_marked_image.png"
        contour_path = contour_output_dir / contour_filename
        cv2.imwrite(str(contour_path), image)

        self.contour_image_path = str(contour_path)
        return str(contour_path)

    def update_averages(self, other_pothole: 'Pothole') -> None:
        """Update averages when merging with another pothole detection.

        Args:
            other_pothole: Another pothole instance to merge with
        """
        # This would be used by the tracking system
        pass
