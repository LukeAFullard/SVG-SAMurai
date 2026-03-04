from lxml import etree
import cairosvg
import io
import base64
from PIL import Image
from typing import Any

# Namespace for SVG creation
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
NSMAP = {None: SVG_NS, "xlink": XLINK_NS}


def create_svg_with_image(image: Image.Image) -> str:
    """Creates a basic SVG string with the given raster image embedded via base64."""
    width, height = image.size
    root = etree.Element(
        "svg",
        width=str(width),
        height=str(height),
        viewBox=f"0 0 {width} {height}",
        nsmap=NSMAP,
    )

    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    image_elem = etree.SubElement(
        root, f"{{{SVG_NS}}}image", width=str(width), height=str(height)
    )
    image_elem.set("href", f"data:image/png;base64,{img_str}")
    image_elem.set(f"{{{XLINK_NS}}}href", f"data:image/png;base64,{img_str}")

    return etree.tostring(root, pretty_print=True, encoding="unicode")


def create_base_svg(width: int, height: int) -> str:
    """Creates a basic empty SVG string with specified dimensions."""
    root = etree.Element(
        "svg",
        width=str(width),
        height=str(height),
        viewBox=f"0 0 {width} {height}",
        nsmap=NSMAP,
    )
    return etree.tostring(root, pretty_print=True, encoding="unicode")


from typing import Union, List, Optional

def add_path_to_svg(
    svg_str: str,
    path_d: Union[str, List[str]],
    path_id: str,
    fill_color: str = "#FF0000",
    opacity: Optional[float] = None,
) -> str:
    """
    Injects an SVG `<path>` into an existing SVG string within a `<g>` group using lxml.
    """
    if not path_d:
        return svg_str

    try:
        # Provide a parser that handles basic errors and mitigates XXE injection securely
        parser = etree.XMLParser(recover=True, no_network=True, resolve_entities=False)
        root = etree.fromstring(
            svg_str.encode("utf-8", errors="replace"), parser=parser
        )
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
    group = etree.SubElement(
        root, f"{{{ns}}}g" if ns else "g", id=path_id, nsmap=new_nsmap
    )

    # Add <title> for tooltips (e.g., Apache eCharts interactivity)
    title = etree.SubElement(group, f"{{{ns}}}title" if ns else "title")
    title.text = path_id

    # Create the <path> elements
    # Using fill-rule="evenodd" is important when combining outer boundaries and inner holes
    if isinstance(path_d, str):
        path_d_list = [path_d]
    else:
        path_d_list = path_d

    for i, pd in enumerate(path_d_list):
        # We can append an index to the id if there are multiple geometries, but
        # since they are in a group with `id=path_id`, we don't strictly need an `id` on each path.
        path_elem = etree.SubElement(
            group,
            f"{{{ns}}}path" if ns else "path",
            d=pd,
            fill=fill_color,
            attrib={"fill-rule": "evenodd"},  # Handles holes properly
        )
        if opacity is not None:
            path_elem.set("opacity", str(opacity))

    return etree.tostring(root, pretty_print=True, encoding="unicode")


def parse_svg_to_image(svg_bytes: bytes) -> Image.Image:
    """Converts uploaded SVG file bytes into a PIL Image."""
    # Pass url_fetcher to block network and local file access from within SVG
    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes, url_fetcher=lambda *args, **kwargs: b""
    )
    return Image.open(io.BytesIO(png_bytes))


def load_image(uploaded_file: Any) -> Image.Image:
    """Loads an uploaded image (Raster or Vector) and returns a PIL Image."""
    if getattr(uploaded_file, "type", "") == "image/svg+xml":
        return parse_svg_to_image(uploaded_file.getvalue())
    else:
        # Handle regular rasters (PNG, JPG)
        return Image.open(uploaded_file).convert("RGB")
