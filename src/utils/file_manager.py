"""File management utilities for the pothole detection system."""

import os
import shutil
import logging
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FileManager:
    """Handles file and directory operations for the project."""

    def __init__(self, base_folder: str = "Extraction"):
        """Initialize the file manager.

        Args:
            base_folder: Base folder for extracted data
        """
        self.base_folder = Path(base_folder)
        self._create_folder(self.base_folder)

    def get_bag_files(self, bag_folder_path: str) -> List[str]:
        """Get list of .bag files in the specified folder.

        Args:
            bag_folder_path: Path to folder containing .bag files

        Returns:
            List of .bag file paths
        """
        bag_path = Path(bag_folder_path)
        return [str(f) for f in bag_path.glob("*.bag") if f.is_file()]

    def get_bag_folders(self) -> List[str]:
        """Get list of bag folders in the base extraction folder.

        Returns:
            List of bag folder paths
        """
        if not self.base_folder.exists():
            return []
        return [str(f) for f in self.base_folder.iterdir() if f.is_dir()]

    def get_images_in_folder(self, folder_path: str) -> List[str]:
        """Get list of image files in the specified folder.

        Args:
            folder_path: Path to folder containing images

        Returns:
            List of image file paths
        """
        images_path = Path(folder_path) / "Imagenes"
        if not images_path.exists():
            logger.error(f"Images directory not found: {images_path}")
            return []

        return [str(f) for f in images_path.glob("*")
                if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

    def get_bag_origin(self, image_path: str, bag_base_path: str) -> str:
        """Get the original .bag file path from an image path.

        Args:
            image_path: Path to the image file
            bag_base_path: Base path where .bag files are located

        Returns:
            Path to the original .bag file
        """
        image_path = Path(image_path)
        # Navigate up two levels to get the bag folder name
        bag_folder = image_path.parent.parent.name
        bag_file = f"{bag_folder}.bag"
        return str(Path(bag_base_path) / bag_file)

    def create_folder(self, folder_path: str) -> None:
        """Create a folder if it doesn't exist.

        Args:
            folder_path: Path to the folder to create
        """
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    def save_image(self, image_path: str, image) -> None:
        """Save an image to the specified path.

        Args:
            image_path: Path where to save the image
            image: Image data (OpenCV format)
        """
        import cv2
        self.create_folder(str(Path(image_path).parent))
        cv2.imwrite(image_path, image)

    def cleanup_extracted_files(self) -> None:
        """Remove all extracted files and folders."""
        if self.base_folder.exists():
            shutil.rmtree(self.base_folder)
            logger.info(f"Cleaned up extraction folder: {self.base_folder}")

    def _create_folder(self, folder_path: str) -> None:
        """Internal method to create folder."""
        Path(folder_path).mkdir(parents=True, exist_ok=True)
