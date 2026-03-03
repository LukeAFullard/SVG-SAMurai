import pytest
from src.xml_manager import create_base_svg, add_path_to_svg, parse_svg_to_image, SVG_NS, NSMAP
from lxml import etree
from unittest.mock import patch, MagicMock
import io
from PIL import Image

def test_create_base_svg():
    svg_str = create_base_svg(800, 600)
    root = etree.fromstring(svg_str.encode('utf-8'))

    assert root.tag == f"{{{SVG_NS}}}svg"
    assert root.attrib["width"] == "800"
    assert root.attrib["height"] == "600"
    assert root.attrib["viewBox"] == "0 0 800 600"

def test_add_path_to_svg():
    svg_str = create_base_svg(100, 100)
    path_d = "M 0,0 L 100,0 L 100,100 L 0,100 Z"
    path_id = "test_square"

    new_svg_str = add_path_to_svg(svg_str, path_d, path_id, fill_color="#00FF00", opacity=0.8)

    # Check that the path string contains the new element
    assert f'id="{path_id}"' in new_svg_str
    assert f'd="{path_d}"' in new_svg_str
    assert 'fill="#00FF00"' in new_svg_str
    assert 'opacity="0.8"' in new_svg_str
    assert 'fill-rule="evenodd"' in new_svg_str

def test_add_path_empty_path():
    svg_str = create_base_svg(100, 100)
    path_d = ""
    new_svg_str = add_path_to_svg(svg_str, path_d, "test")

    # Should be identical to the original SVG since the path was empty
    assert new_svg_str == svg_str

def test_add_path_invalid_xml():
    invalid_xml = "This is not an XML document"
    path_d = "M 0,0 L 100,100"
    result = add_path_to_svg(invalid_xml, path_d, "test")

    # Should gracefully return the original string if parsing fails
    assert result == invalid_xml

def test_parse_svg_to_image():
    """
    Verifies that parse_svg_to_image correctly converts SVG bytes
    to a PIL Image using cairosvg and PIL.Image.open.
    """
    svg_bytes = b"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\"></svg>"
    mock_png_bytes = b"fake png data"

    with patch("cairosvg.svg2png", return_value=mock_png_bytes) as mock_svg2png:
        with patch("PIL.Image.open") as mock_image_open:
            mock_image = MagicMock(spec=Image.Image)
            mock_image_open.return_value = mock_image

            result = parse_svg_to_image(svg_bytes)

            # Verify cairosvg.svg2png was called with the correct bytes
            mock_svg2png.assert_called_once_with(bytestring=svg_bytes)

            # Verify Image.open was called with a BytesIO wrapping the PNG bytes
            mock_image_open.assert_called_once()
            args, _ = mock_image_open.call_args
            assert isinstance(args[0], io.BytesIO)
            assert args[0].getvalue() == mock_png_bytes

            # Verify the result is the expected image object
            assert result == mock_image

@patch("src.xml_manager.parse_svg_to_image")
def test_load_image_svg(mock_parse_svg_to_image):
    """Verifies load_image correctly handles SVG files."""
    mock_file = MagicMock()
    mock_file.type = "image/svg+xml"
    mock_file.getvalue.return_value = b"<svg></svg>"

    mock_image = MagicMock(spec=Image.Image)
    mock_parse_svg_to_image.return_value = mock_image

    from src.xml_manager import load_image
    result = load_image(mock_file)

    mock_parse_svg_to_image.assert_called_once_with(b"<svg></svg>")
    assert result == mock_image

@patch("PIL.Image.open")
def test_load_image_raster(mock_image_open):
    """Verifies load_image correctly handles raster files (PNG, JPG)."""
    mock_file = MagicMock()
    mock_file.type = "image/png"

    mock_image = MagicMock(spec=Image.Image)
    mock_converted_image = MagicMock(spec=Image.Image)
    mock_image.convert.return_value = mock_converted_image
    mock_image_open.return_value = mock_image

    from src.xml_manager import load_image
    result = load_image(mock_file)

    mock_image_open.assert_called_once_with(mock_file)
    mock_image.convert.assert_called_once_with("RGB")
    assert result == mock_converted_image
