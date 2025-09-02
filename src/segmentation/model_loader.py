"""Model loading utilities for segmentation."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading of trained segmentation models."""

    def __init__(self, model_path: str, id2label: Optional[Dict[int, str]] = None):
        """Initialize the model loader.

        Args:
            model_path: Path to the saved model state dict
            id2label: Mapping from class IDs to labels
        """
        self.model_path = Path(model_path)
        self.id2label = id2label or {
            0: "background",
            1: "pothole",
        }

    def load_model(self, model_class: Any) -> Any:
        """Load the trained model.

        Args:
            model_class: The model class to instantiate

        Returns:
            Loaded model ready for inference
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Create model instance
        model = model_class(id2label=self.id2label)

        # Load state dict
        try:
            state_dict = torch.load(self.model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Set to evaluation mode
        model.eval()

        return model
