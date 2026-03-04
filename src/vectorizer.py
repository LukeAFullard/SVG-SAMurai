import cv2
import numpy as np


from typing import List, Union

def mask_to_svg_path(mask: np.ndarray, epsilon_factor: float = 0.005) -> Union[str, List[str]]:
    """
    Converts a binary mask to SVG path strings.

    Args:
        mask (np.ndarray): The 2D binary mask.
        epsilon_factor (float): The factor for approximating the contour with Ramer-Douglas-Peucker algorithm.
            A higher value means more simplification (fewer points, smaller SVG size).

    Returns:
        List[str]: A list of SVG path data strings (`M x,y L x,y Z ...`), one for each external contour.
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

    if contours is None or len(contours) == 0:
        return []

    # 2. Iterate through contours and hierarchy to build the path
    # The hierarchy array has shape (1, num_contours, 4)
    # The 4 elements are: [Next, Previous, First_Child, Parent]
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]

    paths = []

    for i, contour in enumerate(contours):
        # We only want to process the contours if it has at least 3 points
        if len(contour) < 3:
            continue

        # Check if it's an external contour or a hole
        # We can group holes into the same path as their parent external contour
        parent_idx = hierarchy[i][3]
        if parent_idx != -1:
            # It's a hole, it will be added to the parent's path if we are combining them.
            # However, to avoid a bounding box issue with multiple non-connected geometries,
            # we should separate out the geometries.
            # In RETR_CCOMP, external contours have parent == -1.
            # Wait, if we separate them into different paths, holes will be filled if they are
            # in a separate <path> tag. To keep holes as holes, they MUST be in the SAME <path> d string
            # as their bounding external contour.
            pass

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

        path_data = []
        # Add the M command for the first point
        path_data.append(f"M {pts[0][0]},{pts[0][1]}")

        # Add the L commands for the rest
        for x, y in pts[1:]:
            path_data.append(f"L {x},{y}")

        # Close the contour
        path_data.append("Z")

        # Keep track of paths
        # Actually, let's group holes with their parent external boundaries.
        # But for now, to fix the bounding box issue, returning a list of all geometries
        # as separate strings might cause holes to be filled.
        # Let's rebuild the hierarchy correctly: each external contour + its holes is one path string!

        # We'll just do that below.
        pass

    # A better approach to group holes with parents:
    paths_grouped = []

    for i, contour in enumerate(contours):
        parent_idx = hierarchy[i][3]

        # If it's an external contour
        if parent_idx == -1:
            # Gather this contour and all its immediate holes
            component_contours = [contour]

            # Find holes that have this contour as their parent
            for j, hole_contour in enumerate(contours):
                if hierarchy[j][3] == i:
                    component_contours.append(hole_contour)

            # Build the path string for this component
            component_path_data = []
            for c in component_contours:
                if len(c) < 3:
                    continue
                epsilon = epsilon_factor * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                if len(approx) < 3:
                    continue
                pts = approx.reshape(-1, 2)
                component_path_data.append(f"M {pts[0][0]},{pts[0][1]}")
                for x, y in pts[1:]:
                    component_path_data.append(f"L {x},{y}")
                component_path_data.append("Z")

            if component_path_data:
                paths_grouped.append(" ".join(component_path_data))

    return paths_grouped
