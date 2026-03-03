import pytest
from src.xml_manager import create_base_svg, add_path_to_svg, SVG_NS, NSMAP
from lxml import etree

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
