import numpy as np
import cv2
from src.vectorizer import mask_to_svg_path

import pytest


def test_mask_to_svg_path_empty_mask():
    # Test with an empty mask (no contours)
    mask = np.zeros((100, 100), dtype=np.uint8)
    paths = mask_to_svg_path(mask)
    assert paths == [], "Empty mask should result in an empty list"


def test_mask_to_svg_path_invalid_mask():
    with pytest.raises(ValueError, match="Mask must be a 2D numpy array."):
        mask_to_svg_path(None)
    with pytest.raises(ValueError, match="Mask must be a 2D numpy array."):
        mask_to_svg_path(np.zeros((100, 100, 3), dtype=np.uint8))


def test_mask_to_svg_path_simple_square():
    # Test with a simple square mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Draw a 50x50 square in the center
    mask[25:75, 25:75] = 255

    paths = mask_to_svg_path(mask, epsilon_factor=0.001)

    assert len(paths) == 1
    path_str = paths[0]
    # We expect a path that starts with M and has 3 L commands, ending with Z
    assert path_str.startswith("M")
    assert "Z" in path_str
    assert path_str.count("M") == 1
    assert path_str.count("L") == 3  # A square has 4 points total

    # The first point should be one of the corners (e.g., M 25,25)
    assert "25" in path_str or "74" in path_str


def test_mask_to_svg_path_with_hole():
    # Test a square with a hole (donut)
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Outer square
    mask[10:90, 10:90] = 255
    # Inner hole
    mask[30:70, 30:70] = 0

    paths = mask_to_svg_path(mask, epsilon_factor=0.001)

    assert len(paths) == 1
    path_str = paths[0]
    # We should have two contours: one for the outer boundary, one for the inner hole
    # So we expect two "M" commands and two "Z" commands in the same string
    assert path_str.count("M") == 2
    assert path_str.count("Z") == 2


def test_mask_to_svg_path_nested_holes():
    # Test a mask with a nested hierarchy: outer square, inner hole, innermost square
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Outer square
    mask[10:90, 10:90] = 255
    # Inner hole
    mask[20:80, 20:80] = 0
    # Innermost solid
    mask[40:60, 40:60] = 255

    paths = mask_to_svg_path(mask, epsilon_factor=0.001)

    # We expect two "islands" (external contours):
    # 1. The innermost solid (has no holes) -> 1 M, 1 Z
    # 2. The outer square (has 1 hole) -> 2 M, 2 Z
    assert len(paths) == 2

    # Check the segments in the paths
    m_counts = [path.count("M") for path in paths]
    z_counts = [path.count("Z") for path in paths]

    # One path should have 1 segment, the other should have 2 segments
    assert sorted(m_counts) == [1, 2]
    assert sorted(z_counts) == [1, 2]


def test_mask_to_svg_path_simplification():
    # Create a noisy/jagged circle
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)

    # Add some noise to the boundary
    for i in range(10):
        mask[100 + int(48 * np.sin(i)), 100 + int(48 * np.cos(i))] = 0

    # Low epsilon (high detail)
    detailed_paths = mask_to_svg_path(mask, epsilon_factor=0.0001)

    # High epsilon (low detail / simplified)
    simplified_paths = mask_to_svg_path(mask, epsilon_factor=0.1)

    assert len(detailed_paths) > 0 and len(simplified_paths) > 0
    detailed_path = detailed_paths[0]
    simplified_path = simplified_paths[0]
    # The simplified path should have fewer points (fewer 'L' commands)
    assert detailed_path.count("L") > simplified_path.count("L")


def test_mask_to_svg_path_contours_none(mocker):
    # Test when cv2.findContours returns None for contours
    mocker.patch("cv2.findContours", return_value=(None, None))
    mask = np.zeros((100, 100), dtype=np.uint8)

    paths = mask_to_svg_path(mask)
    assert paths == [], "Should return empty list when contours is None"


def test_mask_to_svg_path_hierarchy_none(mocker):
    # Test when cv2.findContours returns valid contours but None for hierarchy
    mocker.patch(
        "cv2.findContours",
        return_value=([np.array([[[0, 0]], [[0, 10]], [[10, 10]]])], None),
    )
    mask = np.zeros((100, 100), dtype=np.uint8)

    paths = mask_to_svg_path(mask)
    assert paths == [], "Should return empty list when hierarchy is None"


def test_mask_to_svg_path_extreme_epsilon():
    # Test with extreme epsilon factors (0.0 and very large) using an existing mask fixture approach
    # We can use a simple square or circle pattern similar to what's done in other tests
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Draw a 50x50 square in the center, same as test_mask_to_svg_path_simple_square
    mask[25:75, 25:75] = 255

    # Epsilon factor 0.0 means no simplification
    paths_zero = mask_to_svg_path(mask, epsilon_factor=0.0)
    assert len(paths_zero) == 1
    # For a square with 0.0 simplification, the number of L commands shouldn't be zero
    assert paths_zero[0].count("L") >= 3

    # Extremely high epsilon factor should simplify the contour so much
    # that it becomes less than 3 points, thus returning an empty list
    paths_huge = mask_to_svg_path(mask, epsilon_factor=100.0)
    assert paths_huge == [], (
        "Extremely high epsilon should over-simplify and return no paths"
    )
