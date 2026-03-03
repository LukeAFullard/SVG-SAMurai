import cv2
import numpy as np

def mask_to_svg_path(mask: np.ndarray, epsilon_factor: float = 0.005) -> str:
    """
    Converts a binary mask to an SVG path string.

    Args:
        mask (np.ndarray): The 2D binary mask.
        epsilon_factor (float): The factor for approximating the contour with Ramer-Douglas-Peucker algorithm.
            A higher value means more simplification (fewer points, smaller SVG size).

    Returns:
        str: An SVG path data string (`M x,y L x,y Z ...`).
    """
    # 1. Extract Contours
    # RETR_CCOMP retrieves all of the contours and organizes them into a two-level hierarchy.
    # At the top level, there are external boundaries of the components.
    # At the second level, there are boundaries of the holes.
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if contours is None or len(contours) == 0:
        return ""

    path_data = []

    # 2. Iterate through contours and hierarchy to build the path
    # The hierarchy array has shape (1, num_contours, 4)
    # The 4 elements are: [Next, Previous, First_Child, Parent]
    if hierarchy is None:
        return ""

    hierarchy = hierarchy[0]

    for i, contour in enumerate(contours):
        # We only want to process the contours if it has at least 3 points
        if len(contour) < 3:
            continue

        # 3. Simplify Contour
        # Calculate epsilon based on the contour's arc length
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # We want to skip highly simplified contours that are just points or lines
        if len(approx) < 3:
            continue

        # 4. Format to SVG path
        # M = moveto (start point)
        # L = lineto (subsequent points)
        # Z = closepath (return to start)
        pts = approx.reshape(-1, 2)

        # Add the M command for the first point
        path_data.append(f"M {pts[0][0]},{pts[0][1]}")

        # Add the L commands for the rest
        for x, y in pts[1:]:
            path_data.append(f"L {x},{y}")

        # Close the contour
        path_data.append("Z")

    return " ".join(path_data)
