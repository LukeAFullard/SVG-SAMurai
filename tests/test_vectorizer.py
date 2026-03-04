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
