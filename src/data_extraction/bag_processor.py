"""Data extraction from Intel RealSense .bag files."""

import os
import threading
import logging
from pathlib import Path
from typing import List, Optional
import pyrealsense2 as rs
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class BagProcessor:
    """Processes Intel RealSense .bag files to extract images and depth data."""

    def __init__(self, bag_files_path: str):
        """Initialize the bag processor.

        Args:
            bag_files_path: Path to folder containing .bag files
        """
        self.bag_files_path = Path(bag_files_path)
        self.bag_files = self._get_bag_files()

    def _get_bag_files(self) -> List[str]:
        """Get list of .bag files in the specified directory."""
        return [str(f) for f in self.bag_files_path.glob("*.bag") if f.is_file()]

    def process_all_bags(self, output_base_folder: str) -> None:
        """Process all .bag files using multithreading.

        Args:
            output_base_folder: Base folder for extracted data
        """
        threads = []
        for bag_file in self.bag_files:
            thread = threading.Thread(
                target=self._process_single_bag,
                args=(bag_file, output_base_folder)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        logger.info("All bag files processed successfully")

    def _process_single_bag(self, bag_file_path: str, output_base_folder: str) -> None:
        """Process a single .bag file.

        Args:
            bag_file_path: Path to the .bag file
            output_base_folder: Base folder for output
        """
        bag_extractor = BagExtractor(bag_file_path, output_base_folder)
        bag_extractor.extract_data()


class BagExtractor:
    """Extracts data from a single .bag file."""

    def __init__(self, bag_file_path: str, base_output_folder: str):
        """Initialize the bag extractor.

        Args:
            bag_file_path: Path to the .bag file
            base_output_folder: Base folder for extracted data
        """
        self.bag_file_path = bag_file_path
        self.base_output_folder = Path(base_output_folder)

        # Create output folders
        bag_name = Path(bag_file_path).stem
        self.images_folder = self.base_output_folder / bag_name / "Imagenes"
        self.depth_folder = self.base_output_folder / bag_name / "ImagenesProfundidad"
        self.ply_folder = self.base_output_folder / bag_name / "Ply"

        self._create_output_folders()

        # Initialize RealSense objects
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()

    def _create_output_folders(self) -> None:
        """Create necessary output folders."""
        for folder in [self.images_folder, self.depth_folder, self.ply_folder]:
            folder.mkdir(parents=True, exist_ok=True)

    def extract_data(self) -> None:
        """Extract all data from the .bag file."""
        pipeline = self._configure_pipeline()
        frame_number = 0

        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                self._save_color_image(color_frame, frame_number)
                self._save_depth_image(depth_frame, frame_number)
                # Optional: save point cloud
                # self._save_point_cloud(depth_frame, aligned_frames, frame_number)

                frame_number += 1

        except RuntimeError:
            logger.info(f"Finished processing {self.bag_file_path}")
        finally:
            pipeline.stop()

    def _configure_pipeline(self) -> rs.pipeline:
        """Configure the RealSense pipeline for playback.

        Returns:
            Configured pipeline object
        """
        config = rs.config()
        config.enable_device_from_file(str(self.bag_file_path), repeat_playback=False)

        pipeline = rs.pipeline()
        pipeline.start(config)

        # Configure for non-real-time playback
        playback = pipeline.get_active_profile().get_device().as_playback()
        playback.set_real_time(False)

        return pipeline

    def _save_color_image(self, color_frame: rs.frame, frame_number: int) -> None:
        """Save color image from frame.

        Args:
            color_frame: RealSense color frame
            frame_number: Frame number for filename
        """
        color_image = np.asanyarray(color_frame.get_data())
        # Convert BGR to RGB for consistency with OpenCV
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        filename = f"frame_{frame_number:05d}.png"
        filepath = self.images_folder / filename
        cv2.imwrite(str(filepath), color_image)

    def _save_depth_image(self, depth_frame: rs.frame, frame_number: int) -> None:
        """Save depth image from frame.

        Args:
            depth_frame: RealSense depth frame
            frame_number: Frame number for filename
        """
        depth_image = np.asanyarray(depth_frame.get_data())

        filename = f"frame_{frame_number:05d}.png"
        filepath = self.depth_folder / filename
        cv2.imwrite(str(filepath), depth_image)

    def _save_point_cloud(self, depth_frame: rs.frame,
                         aligned_frames: rs.frameset,
                         frame_number: int) -> None:
        """Save point cloud as PLY file.

        Args:
            depth_frame: RealSense depth frame
            aligned_frames: Aligned frameset
            frame_number: Frame number for filename
        """
        self.pc.map_to(aligned_frames.get_color_frame())
        points = self.pc.calculate(depth_frame)

        filename = f"frame_{frame_number:05d}.ply"
        filepath = self.ply_folder / filename
        points.export_to_ply(str(filepath), aligned_frames.get_color_frame())
