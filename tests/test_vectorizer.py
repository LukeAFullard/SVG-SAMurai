import numpy as np
import cv2
from src.vectorizer import mask_to_svg_path

def test_mask_to_svg_path_empty_mask():
    # Test with an empty mask (no contours)
    mask = np.zeros((100, 100), dtype=np.uint8)
    path_str = mask_to_svg_path(mask)
    assert path_str == "", "Empty mask should result in an empty string"

def test_mask_to_svg_path_simple_square():
    # Test with a simple square mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Draw a 50x50 square in the center
    mask[25:75, 25:75] = 255

    path_str = mask_to_svg_path(mask, epsilon_factor=0.001)

    # We expect a path that starts with M and has 3 L commands, ending with Z
    assert path_str.startswith("M")
    assert "Z" in path_str
    assert path_str.count("M") == 1
    assert path_str.count("L") == 3 # A square has 4 points total

    # The first point should be one of the corners (e.g., M 25,25)
    assert "25" in path_str or "74" in path_str

def test_mask_to_svg_path_with_hole():
    # Test a square with a hole (donut)
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Outer square
    mask[10:90, 10:90] = 255
    # Inner hole
    mask[30:70, 30:70] = 0

    path_str = mask_to_svg_path(mask, epsilon_factor=0.001)

    # We should have two contours: one for the outer boundary, one for the inner hole
    # So we expect two "M" commands and two "Z" commands
    assert path_str.count("M") == 2
    assert path_str.count("Z") == 2

def test_mask_to_svg_path_simplification():
    # Create a noisy/jagged circle
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)

    # Add some noise to the boundary
    for i in range(10):
        mask[100+int(48*np.sin(i)), 100+int(48*np.cos(i))] = 0

    # Low epsilon (high detail)
    detailed_path = mask_to_svg_path(mask, epsilon_factor=0.0001)

    # High epsilon (low detail / simplified)
    simplified_path = mask_to_svg_path(mask, epsilon_factor=0.1)

    # The simplified path should have fewer points (fewer 'L' commands)
    assert detailed_path.count("L") > simplified_path.count("L")

def test_mask_to_svg_path_contours_none(mocker):
    # Test when cv2.findContours returns None for contours
    mocker.patch('cv2.findContours', return_value=(None, None))
    mask = np.zeros((100, 100), dtype=np.uint8)

    path_str = mask_to_svg_path(mask)
    assert path_str == "", "Should return empty string when contours is None"

def test_mask_to_svg_path_hierarchy_none(mocker):
    # Test when cv2.findContours returns valid contours but None for hierarchy
    mocker.patch('cv2.findContours', return_value=([np.array([[[0, 0]], [[0, 10]], [[10, 10]]])], None))
    mask = np.zeros((100, 100), dtype=np.uint8)

    path_str = mask_to_svg_path(mask)
    assert path_str == "", "Should return empty string when hierarchy is None"
