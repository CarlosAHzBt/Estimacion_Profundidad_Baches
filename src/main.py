"""Main entry point for the pothole detection system."""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_extraction.bag_processor import BagProcessor
from segmentation.model_loader import ModelLoader
from segmentation.segmenter import PotholeSegmenter
from analysis.pothole import Pothole
from tracking.tracker import PotholeTracker
from utils.config import Config
from utils.file_manager import FileManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PotholeDetectionSystem:
    """Main system for pothole detection and analysis."""

    def __init__(self, bag_folder: str, output_folder: str, config: Optional[Config] = None):
        """Initialize the detection system.

        Args:
            bag_folder: Path to folder containing .bag files
            output_folder: Path for output results
            config: System configuration
        """
        self.bag_folder = Path(bag_folder)
        self.output_folder = Path(output_folder)
        self.config = config or Config()

        # Initialize components
        self.file_manager = FileManager()
        self.bag_processor = BagProcessor(str(self.bag_folder))
        self.model_loader = ModelLoader(self.config.model_path)
        self.segmenter = None
        self.tracker = PotholeTracker(
            position_tolerance=self.config.position_tolerance,
            size_tolerance=self.config.size_tolerance,
            temporal_tolerance=self.config.temporal_tolerance
        )

        self.detected_potholes: List[Pothole] = []

    def run(self) -> None:
        """Run the complete pothole detection pipeline."""
        logger.info("Starting pothole detection system")

        self._extract_data()
        self._load_model()
        self._process_images()
        self._analyze_potholes()
        self._group_potholes()
        self._generate_report()

        logger.info("Pothole detection completed")

    def _extract_data(self) -> None:
        """Extract data from .bag files."""
        logger.info("Extracting data from .bag files")
        self.bag_processor.process_all_bags(str(self.output_folder / self.config.base_extraction_folder))

    def _load_model(self) -> None:
        """Load the segmentation model."""
        logger.info("Loading segmentation model")
        # Import here to avoid circular imports
        sys.path.append(str(Path(__file__).parent.parent / "RutaModelo"))
        from Segformer_FineTuner import SegformerFinetuner

        model = self.model_loader.load_model(SegformerFinetuner)
        self.segmenter = PotholeSegmenter(
            model=model,
            min_area=self.config.min_pothole_area,
            device=self.config.device
        )

    def _process_images(self) -> None:
        """Process images and detect potholes."""
        logger.info("Processing images for pothole detection")

        # Get all bag folders
        bag_folders = self.file_manager.get_bag_folders()
        if not bag_folders:
            logger.warning("No bag folders found for processing")
            return

        # Process each bag folder
        for bag_folder in bag_folders:
            bag_path = Path(self.config.base_extraction_folder) / bag_folder
            self._process_bag_folder(bag_path)

    def _process_bag_folder(self, bag_path: Path) -> None:
        """Process all images in a bag folder.

        Args:
            bag_path: Path to the bag folder
        """
        image_files = self.file_manager.get_images_in_folder(str(bag_path))

        # Process images in batches
        for i in range(0, len(image_files), self.config.batch_size):
            batch = image_files[i:i + self.config.batch_size]
            self._process_image_batch(bag_path, batch)

    def _process_image_batch(self, bag_path: Path, image_batch: List[str]) -> None:
        """Process a batch of images.

        Args:
            bag_path: Path to the bag folder
            image_batch: List of image file paths
        """
        with ThreadPoolExecutor() as executor:
            future_to_image = {}

            for image_path in image_batch:
                future = executor.submit(self._process_single_image, bag_path, image_path)
                future_to_image[future] = image_path

            for future in as_completed(future_to_image):
                try:
                    potholes = future.result()
                    self.detected_potholes.extend(potholes)
                except Exception as exc:
                    image_path = future_to_image[future]
                    logger.error(f"Error processing image {image_path}: {exc}")

    def _process_single_image(self, bag_path: Path, image_path: str) -> List[Pothole]:
        """Process a single image for pothole detection.

        Args:
            bag_path: Path to the bag folder
            image_path: Path to the image file

        Returns:
            List of detected potholes
        """
        # Run segmentation
        pothole_coords = self.segmenter.segment_image(image_path)

        potholes = []
        for i, coords in enumerate(pothole_coords):
            pothole_id = f"{Path(image_path).stem}_{i}"
            bag_origin_path = self.file_manager.get_bag_origin(
                image_path, str(self.bag_folder)
            )
            bag_name = Path(bag_origin_path).stem

            pothole = Pothole(
                bag_origin=bag_name,
                bag_path=bag_origin_path,  # Path to the specific .bag file
                output_folder=str(self.output_folder),
                rgb_image_path=image_path,
                pothole_id=pothole_id,
                coordinates=coords,
                min_diameter_mm=self.config.min_diameter_mm
            )

            potholes.append(pothole)

        return potholes

    def _analyze_potholes(self) -> None:
        """Analyze detected potholes."""
        logger.info("Analyzing detected potholes")

        valid_potholes = []
        with ThreadPoolExecutor() as executor:
            future_to_pothole = {
                executor.submit(pothole.process_pothole): pothole
                for pothole in self.detected_potholes
            }

            for future in as_completed(future_to_pothole):
                try:
                    success = future.result()
                    if success:
                        valid_potholes.append(future_to_pothole[future])
                except Exception as exc:
                    logger.error(f"Error analyzing pothole: {exc}")

        self.detected_potholes = valid_potholes
        logger.info(f"Successfully analyzed {len(valid_potholes)} potholes")

    def _group_potholes(self) -> None:
        """Group similar potholes across frames."""
        logger.info("Grouping similar potholes")

        # Add all potholes to tracker
        for pothole in self.detected_potholes:
            self.tracker.add_pothole(pothole)

        # Group and calculate averages
        groups = self.tracker.group_potholes()
        self.detected_potholes = self.tracker.calculate_averages(groups)

        logger.info(f"Grouped into {len(self.detected_potholes)} unique potholes")

    def _generate_report(self) -> None:
        """Generate CSV report of detected potholes."""
        logger.info("Generating detection report")

        # Sort potholes
        sorted_potholes = sorted(
            self.detected_potholes,
            key=lambda p: (p.bag_origin, p.pothole_id)
        )

        # Create output file
        output_file = self.output_folder / 'potholes_report.csv'
        self.output_folder.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Pothole ID",
                "Average Max Radius (mm)",
                "Average Depth (m)",
                "Contour Image",
                "Grouped Frames"
            ])

            grouped_ids = set()
            for pothole in sorted_potholes:
                if pothole.pothole_id not in grouped_ids:
                    # Mark all grouped frames as processed
                    grouped_ids.update(pothole.grouped_frames)

                    writer.writerow([
                        pothole.pothole_id,
                        f"{pothole.max_radius_mm:.2f}" if pothole.max_radius_mm else "N/A",
                        f"{pothole.estimated_depth:.3f}" if pothole.estimated_depth else "N/A",
                        pothole.contour_image_path if hasattr(pothole, 'contour_image_path') else "N/A",
                        ', '.join(pothole.grouped_frames)
                    ])

        logger.info(f"Report generated: {output_file}")


def main():
    """Main entry point with GUI for folder selection."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        # Select bag folder
        bag_folder = filedialog.askdirectory(title="Select folder containing .bag files")
        if not bag_folder:
            logger.error("No bag folder selected")
            return

        # Select output folder
        output_folder = filedialog.askdirectory(title="Select output folder")
        if not output_folder:
            logger.error("No output folder selected")
            return

        # Run detection system
        config = Config()
        system = PotholeDetectionSystem(bag_folder, output_folder, config)
        system.run()

        logger.info("Processing completed successfully")

    except ImportError:
        logger.error("Tkinter not available for GUI. Please provide paths as arguments.")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
