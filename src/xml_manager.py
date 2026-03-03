import os
from lxml import etree
import cairosvg
import io
from PIL import Image

# Namespace for SVG creation
SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {None: SVG_NS}

def create_base_svg(width: int, height: int) -> str:
    """Creates a basic empty SVG string with specified dimensions."""
    root = etree.Element("svg", width=str(width), height=str(height), viewBox=f"0 0 {width} {height}", nsmap=NSMAP)
    return etree.tostring(root, pretty_print=True, encoding="unicode")

def add_path_to_svg(svg_str: str, path_d: str, path_id: str, fill_color: str = "#FF0000", opacity: float = 0.5) -> str:
    """
    Injects an SVG `<path>` into an existing SVG string within a `<g>` group using lxml.
    """
    if not path_d:
        return svg_str

    try:
        # Provide a parser that handles basic errors and mitigates XXE injection
        parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
        root = etree.fromstring(svg_str.encode('utf-8', errors='replace'), parser=parser)
        if root is None:
            return svg_str
    except Exception:
        # If the string isn't an XML document or parsing fails
        return svg_str

    # Find the correct namespace for the root or default to SVG_NS
    ns = SVG_NS
    if root.nsmap and None in root.nsmap:
        ns = root.nsmap[None]
    elif root.tag.startswith("{"):
        ns = root.tag[1:].split("}")[0]

    # Clean the namespace map to avoid redundant ns0 prefixes
    # Ensure xmlns is explicitly available in nsmap of new elements
    new_nsmap = {None: ns} if ns else None

    # Create the <g id="path_id">
    group = etree.SubElement(root, f"{{{ns}}}g" if ns else "g", id=path_id, nsmap=new_nsmap)

    # Create the <path>
    # Using fill-rule="evenodd" is important when combining outer boundaries and inner holes
    path = etree.SubElement(
        group,
        f"{{{ns}}}path" if ns else "path",
        d=path_d,
        fill=fill_color,
        opacity=str(opacity),
        attrib={"fill-rule": "evenodd"} # Handles holes properly
    )

    return etree.tostring(root, pretty_print=True, encoding="unicode")

def svg_to_png_bytes(svg_str: str) -> bytes:
    """Converts an SVG string into PNG bytes using cairosvg."""
    return cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))

def parse_svg_to_image(svg_bytes: bytes) -> Image.Image:
    """Converts uploaded SVG file bytes into a PIL Image."""
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes)
    return Image.open(io.BytesIO(png_bytes))

def load_image(uploaded_file) -> Image.Image:
    """Loads an uploaded image (Raster or Vector) and returns a PIL Image."""
    if uploaded_file.type == "image/svg+xml":
        return parse_svg_to_image(uploaded_file.getvalue())
    else:
        # Handle regular rasters (PNG, JPG)
        return Image.open(uploaded_file).convert("RGB")
