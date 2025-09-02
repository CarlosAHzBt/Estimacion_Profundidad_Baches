"""Configuration settings for the pothole detection project."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for the pothole detection system."""

    # Model settings
    model_path: str = "model_state_dictV18-este_ya_trae_ruido.pth"
    min_pothole_area: int = 3000
    image_shape: tuple[int, int] = (480, 848)

    # Processing settings
    batch_size: int = 20
    min_diameter_mm: float = 80.0

    # Tracking settings
    position_tolerance: int = 200
    size_tolerance: int = 50
    temporal_tolerance: int = 3

    # Output settings
    base_extraction_folder: str = "Extraction"
    results_folder: str = "Results"

    # Device settings
    device: Optional[str] = None

    def __post_init__(self):
        """Set device if not specified."""
        if self.device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables."""
        return cls(
            model_path=os.getenv('MODEL_PATH', cls.model_path),
            min_pothole_area=int(os.getenv('MIN_POTHOLE_AREA', cls.min_pothole_area)),
            batch_size=int(os.getenv('BATCH_SIZE', cls.batch_size)),
            position_tolerance=int(os.getenv('POSITION_TOLERANCE', cls.position_tolerance)),
            size_tolerance=int(os.getenv('SIZE_TOLERANCE', cls.size_tolerance)),
            temporal_tolerance=int(os.getenv('TEMPORAL_TOLERANCE', cls.temporal_tolerance)),
        )
