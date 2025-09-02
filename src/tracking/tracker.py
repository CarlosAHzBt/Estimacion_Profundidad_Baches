"""Pothole tracking and grouping across frames."""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from ..analysis.pothole import Pothole

logger = logging.getLogger(__name__)


class PotholeTracker:
    """Tracks and groups pothole detections across multiple frames."""

    def __init__(self,
                 position_tolerance: int = 200,
                 size_tolerance: int = 50,
                 temporal_tolerance: int = 3):
        """Initialize the pothole tracker.

        Args:
            position_tolerance: Maximum pixel distance for grouping
            size_tolerance: Maximum size difference for grouping
            temporal_tolerance: Maximum frame gap for grouping
        """
        self.position_tolerance = position_tolerance
        self.size_tolerance = size_tolerance
        self.temporal_tolerance = temporal_tolerance
        self.potholes_history: List[Pothole] = []
        self.processed_potholes = set()

    def add_pothole(self, pothole: Pothole) -> None:
        """Add a pothole to the tracking history.

        Args:
            pothole: Pothole instance to add
        """
        self.potholes_history.append(pothole)

    def group_potholes(self) -> List[List[Pothole]]:
        """Group similar potholes across frames.

        Returns:
            List of pothole groups
        """
        groups = []
        self.processed_potholes = set()
        potholes_by_frame = self._organize_potholes_by_frame()

        for current_frame, current_potholes in potholes_by_frame.items():
            if current_frame == 0:
                continue

            previous_potholes = potholes_by_frame.get(current_frame - 1, [])
            cost_matrix = self._create_cost_matrix(previous_potholes, current_potholes)

            for current_pothole, previous_pothole in cost_matrix.items():
                if previous_pothole and self._are_same_pothole(current_pothole, previous_pothole):
                    group = self._find_group(groups, previous_pothole)
                    if group:
                        group.append(current_pothole)
                        self._log_grouping(current_pothole, group)
                    else:
                        groups.append([previous_pothole, current_pothole])
                    self.processed_potholes.update(current_pothole.grouped_frames)
                else:
                    # Create new group for ungrouped pothole
                    new_group = [current_pothole]
                    groups.append(new_group)

        return groups

    def _organize_potholes_by_frame(self) -> Dict[int, List[Pothole]]:
        """Organize potholes by frame number.

        Returns:
            Dictionary mapping frame numbers to pothole lists
        """
        potholes_by_frame = {}
        for pothole in self.potholes_history:
            frame_num = self._extract_frame_number(pothole.pothole_id)
            if frame_num not in potholes_by_frame:
                potholes_by_frame[frame_num] = []
            potholes_by_frame[frame_num].append(pothole)

        # Sort potholes in each frame by ID
        for frame_num in potholes_by_frame:
            potholes_by_frame[frame_num].sort(key=lambda p: p.pothole_id)

        return dict(sorted(potholes_by_frame.items()))

    def _create_cost_matrix(self, previous_potholes: List[Pothole],
                           current_potholes: List[Pothole]) -> Dict[Pothole, Optional[Pothole]]:
        """Create cost matrix for pothole matching.

        Args:
            previous_potholes: Potholes from previous frame
            current_potholes: Potholes from current frame

        Returns:
            Dictionary mapping current potholes to best previous matches
        """
        cost_matrix = {}
        for current_pothole in current_potholes:
            best_previous = None
            min_distance = float('inf')

            for previous_pothole in previous_potholes:
                distance = np.linalg.norm(
                    np.array(current_pothole.center_circle) -
                    np.array(previous_pothole.center_circle)
                )

                logger.debug(f'Comparing {current_pothole.pothole_id} with '
                           f'{previous_pothole.pothole_id}: Distance={distance}')

                if distance < min_distance and distance < self.position_tolerance:
                    min_distance = distance
                    best_previous = previous_pothole

            cost_matrix[current_pothole] = best_previous
            logger.debug(f'Association: {current_pothole.pothole_id} -> '
                       f'{best_previous.pothole_id if best_previous else "None"} '
                       f'(Distance: {min_distance})')

        return cost_matrix

    def _find_group(self, groups: List[List[Pothole]], pothole: Pothole) -> Optional[List[Pothole]]:
        """Find the group containing a specific pothole.

        Args:
            groups: List of pothole groups
            pothole: Pothole to find

        Returns:
            Group containing the pothole or None
        """
        for group in groups:
            if pothole in group:
                return group
        return None

    def _are_same_pothole(self, pothole1: Pothole, pothole2: Pothole) -> bool:
        """Check if two potholes are the same based on criteria.

        Args:
            pothole1: First pothole
            pothole2: Second pothole

        Returns:
            True if they are considered the same pothole
        """
        if pothole2 is None:
            return False

        # Position check
        distance = np.linalg.norm(
            np.array(pothole1.center_circle) - np.array(pothole2.center_circle)
        )

        # Size check
        radius_diff = abs(pothole1.circle_radius_pixels - pothole2.circle_radius_pixels)

        # Temporal check
        temporal_diff = abs(
            self._extract_frame_number(pothole1.pothole_id) -
            self._extract_frame_number(pothole2.pothole_id)
        )

        logger.debug(f'Comparing: {pothole1.pothole_id} with {pothole2.pothole_id} '
                   f'(Distance: {distance}, Radius diff: {radius_diff}, '
                   f'Temporal diff: {temporal_diff})')

        return (
            distance < self.position_tolerance and
            radius_diff < self.size_tolerance and
            temporal_diff <= self.temporal_tolerance
        )

    def _extract_frame_number(self, pothole_id: str) -> int:
        """Extract frame number from pothole ID.

        Args:
            pothole_id: Pothole identifier

        Returns:
            Frame number
        """
        try:
            return int(pothole_id.split('_')[1])
        except (IndexError, ValueError):
            return 0

    def _log_grouping(self, pothole: Pothole, group: List[Pothole]) -> None:
        """Log pothole grouping information.

        Args:
            pothole: Pothole being grouped
            group: Group it was added to
        """
        existing_ids = [p.pothole_id for p in group]
        logger.info(f'Pothole {pothole.pothole_id} grouped with existing potholes: '
                   f'{", ".join(existing_ids)}')

    def calculate_averages(self, groups: List[List[Pothole]]) -> List[Pothole]:
        """Calculate average properties for grouped potholes.

        Args:
            groups: List of pothole groups

        Returns:
            List of averaged pothole representatives
        """
        averaged_potholes = []

        for group in groups:
            if not group:
                continue

            # Use first pothole as representative
            representative = group[0]

            if len(group) > 1:
                # Calculate averages
                avg_radius = np.mean([p.max_radius_mm for p in group if p.max_radius_mm])
                avg_depth = np.mean([p.estimated_depth for p in group if p.estimated_depth])

                representative.max_radius_mm = avg_radius
                representative.estimated_depth = avg_depth

            # Update grouped frames
            representative.grouped_frames = [p.pothole_id for p in group]

            averaged_potholes.append(representative)

        return averaged_potholes
