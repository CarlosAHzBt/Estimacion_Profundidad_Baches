"""Image segmentation for pothole detection."""

import logging
from pathlib import Path
from typing import List, Tuple, Any
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from skimage.measure import label, regionprops
from skimage.transform import resize
import numpy as np

logger = logging.getLogger(__name__)


class PotholeSegmenter:
    """Handles image segmentation for pothole detection."""

    def __init__(self, model: Any, min_area: int = 3000, device: str = "cpu"):
        """Initialize the segmenter.

        Args:
            model: Trained segmentation model
            min_area: Minimum area for valid pothole detection
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.device = device
        self.min_area = min_area

        # Define preprocessing transforms
        self.transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def segment_image(self, image_path: str) -> List[np.ndarray]:
        """Segment potholes in an image.

        Args:
            image_path: Path to the input image

        Returns:
            List of coordinate arrays for detected potholes
        """
        # Preprocess image
        pixel_values = self._preprocess_image(image_path)

        # Run inference
        predicted_mask = self._run_inference(pixel_values)

        # Post-process mask
        mask_resized = self._resize_mask(predicted_mask)
        labeled_mask = self._label_regions(mask_resized)
        pothole_coords = self._filter_regions(labeled_mask)

        return pothole_coords

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed tensor
        """
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transforms(image).unsqueeze(0).to(self.device)
        return pixel_values

    def _run_inference(self, pixel_values: torch.Tensor) -> np.ndarray:
        """Run model inference.

        Args:
            pixel_values: Preprocessed image tensor

        Returns:
            Predicted segmentation mask
        """
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            # Get mask for pothole class (assuming class 1 is pothole)
            predicted_mask = (outputs[0].argmax(dim=1) == 1).squeeze().cpu().numpy().astype(int)

        return predicted_mask

    def _resize_mask(self, mask: np.ndarray, target_shape: Tuple[int, int] = (480, 848)) -> np.ndarray:
        """Resize mask to target shape.

        Args:
            mask: Input mask
            target_shape: Target dimensions

        Returns:
            Resized mask
        """
        resized = resize(mask, target_shape, order=0,
                        preserve_range=True, anti_aliasing=False).astype(int)
        return resized

    def _label_regions(self, mask: np.ndarray) -> np.ndarray:
        """Label connected regions in mask.

        Args:
            mask: Binary mask

        Returns:
            Labeled mask
        """
        return label(mask, connectivity=2)

    def _filter_regions(self, labeled_mask: np.ndarray) -> List[np.ndarray]:
        """Filter regions by minimum area.

        Args:
            labeled_mask: Labeled mask from regionprops

        Returns:
            List of coordinate arrays for valid regions
        """
        regions = regionprops(labeled_mask)
        valid_coords = []

        for region in regions:
            if region.area >= self.min_area:
                valid_coords.append(region.coords)

        return valid_coords
