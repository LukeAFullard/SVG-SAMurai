import cv2
import numpy as np
from typing import List, Optional


def mask_to_svg_path(mask: np.ndarray, epsilon_factor: float = 0.005) -> List[str]:
    """
    Converts a binary mask to SVG path strings, grouping holes with their parent boundaries.

    Args:
        mask (np.ndarray): The 2D binary mask.
        epsilon_factor (float): The factor for approximating the contour with Ramer-Douglas-Peucker algorithm.
            A higher value means more simplification (fewer points, smaller SVG size).

    Returns:
        List[str]: A list of SVG path data strings (`M x,y L x,y Z ...`), one for each 'island'
                  (external contour and its associated holes).
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("Mask must be a 2D numpy array.")

    # 1. Extract Contours
    # RETR_CCOMP retrieves all of the contours and organizes them into a two-level hierarchy.
    # At the top level, there are external boundaries of the components.
    # At the second level, there are boundaries of the holes.
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours is None or len(contours) == 0 or hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    paths_grouped = []

    # 2. Iterate through contours and hierarchy to build the path
    # The hierarchy array elements are: [Next, Previous, First_Child, Parent]
    for i, contour in enumerate(contours):
        # We only want to process external contours at the top level (Parent == -1)
        if hierarchy[i][3] != -1:
            continue

        # Build the path segments for this "island" (external boundary + holes)
        island_segments = []

        # Process external contour
        ext_path = _contour_to_path_segment(contour, epsilon_factor)
        if ext_path:
            island_segments.append(ext_path)

            # Find holes that have this contour as their parent
            # In RETR_CCOMP, holes are the immediate children of the external contour.
            # We follow the 'First_Child' then 'Next' pointers.
            hole_idx = hierarchy[i][2]
            while hole_idx != -1:
                hole_path = _contour_to_path_segment(contours[hole_idx], epsilon_factor)
                if hole_path:
                    island_segments.append(hole_path)
                hole_idx = hierarchy[hole_idx][0]  # Move to next hole at same level

        if island_segments:
            # Combine all segments into one path string.
            # When multiple M...Z segments are in one 'd' attribute,
            # SVG uses the fill-rule (usually evenodd) to determine what's a hole.
            paths_grouped.append(" ".join(island_segments))

    return paths_grouped


def _contour_to_path_segment(
    contour: np.ndarray, epsilon_factor: float
) -> Optional[str]:
    """Helper to simplify a contour and format it as an SVG path segment (M...Z)."""
    if len(contour) < 3:
        return None

    # Simplify Contour
    # Calculate epsilon based on the contour's arc length
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # We want to skip highly simplified contours that are just points or lines
    if len(approx) < 3:
        return None

    # Format to SVG path segment
    # M = moveto, L = lineto, Z = closepath
    pts = approx.reshape(-1, 2)

    path_data = []
    path_data.append(f"M {pts[0][0]},{pts[0][1]}")
    for x, y in pts[1:]:
        path_data.append(f"L {x},{y}")
    path_data.append("Z")

    return " ".join(path_data)
