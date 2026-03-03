import streamlit as st
from PIL import Image
import numpy as np
import io
import time
from streamlit_image_coordinates import streamlit_image_coordinates

from src.model import compute_image_embedding, predict_mask
from src.vectorizer import mask_to_svg_path
from src.xml_manager import SVG_NS, NSMAP, load_image

# Constants for caching and visual elements
MAX_IMAGE_WIDTH = 800

st.set_page_config(page_title="SVG-SAMurai", layout="wide", page_icon="🗡️")

# Session State Initialization
if "image" not in st.session_state:
    st.session_state.image = None
if "image_embedding" not in st.session_state:
    st.session_state.image_embedding = None
if "points" not in st.session_state:
    st.session_state.points = []
if "labels" not in st.session_state:
    st.session_state.labels = []
if "current_mask" not in st.session_state:
    st.session_state.current_mask = None
if "segments" not in st.session_state:
    st.session_state.segments = {}
if "original_svg" not in st.session_state:
    st.session_state.original_svg = None

st.title("SVG-SAMurai 🗡️")
st.markdown("Transform raster and vector images into segmented SVG paths using the **Segment Anything Model (SAM)**.")

# File uploader
uploaded_file = st.file_uploader("Upload an Image (PNG, JPG, SVG)", type=["png", "jpg", "jpeg", "svg"])

if uploaded_file is not None:
    # Reset state if a new file is uploaded
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.image = None
        st.session_state.image_embedding = None
        st.session_state.points = []
        st.session_state.labels = []
        st.session_state.current_mask = None
        st.session_state.segments = {}
        st.session_state.original_svg = None

    if st.session_state.image is None:
        # Load the image
        with st.spinner("Processing Image..."):
            image = load_image(uploaded_file)
            st.session_state.image = image

            # If the original file was an SVG, save its string representation
            if uploaded_file.type == "image/svg+xml":
                st.session_state.original_svg = uploaded_file.getvalue().decode('utf-8')
            else:
                # Create a blank SVG canvas with the original raster image dimensions
                width, height = image.size
                st.session_state.original_svg = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="{SVG_NS}"></svg>'

            # Compute image embeddings once
            st.session_state.image_embedding = compute_image_embedding(image)
            st.success("Image embedded successfully!")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Interactive Segmentation")

        # Display the image with coordinates clicker
        # If there's a mask, we overlay it
        display_image = st.session_state.image.copy()
        if st.session_state.current_mask is not None:
            # Create a semi-transparent blue overlay for the current mask
            overlay = np.zeros((*st.session_state.current_mask.shape, 4), dtype=np.uint8)
            overlay[st.session_state.current_mask > 0] = [0, 0, 255, 128] # Blue, 50% opacity
            overlay_image = Image.fromarray(overlay, mode="RGBA")
            display_image = display_image.convert("RGBA")
            display_image.paste(overlay_image, (0, 0), overlay_image)
            display_image = display_image.convert("RGB") # Convert back to RGB for display

        # Show the image using streamlit-image-coordinates
        # Note: we need to handle scaling if the image is wider than the container
        # streamlit-image-coordinates scales the image to the container width but gives the
        # coordinates relative to the original image dimensions.
        value = streamlit_image_coordinates(display_image, key="image_coord")

        # Handle clicks
        if value is not None:
            # streamlit_image_coordinates returns x, y relative to the original image size
            x, y = value["x"], value["y"]

            # Determine if it's a positive or negative prompt
            # For simplicity, let's say left click is positive, and we can add a toggle for negative
            is_positive = st.sidebar.checkbox("Next Click is Negative Prompt (Exclude)", value=False, key="neg_prompt_toggle")
            label = 0 if is_positive else 1

            # Check if this is a new click (prevent reruns from adding the same point repeatedly)
            new_point = [x, y]
            if not st.session_state.points or st.session_state.points[-1] != new_point:
                st.session_state.points.append(new_point)
                st.session_state.labels.append(label)

                # Predict new mask
                with st.spinner("Predicting Segment..."):
                    mask = predict_mask(
                        st.session_state.image,
                        st.session_state.image_embedding,
                        st.session_state.points,
                        st.session_state.labels
                    )
                    st.session_state.current_mask = mask
                st.rerun()

        # Tools for interacting with the points
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Undo Last Click"):
                if st.session_state.points:
                    st.session_state.points.pop()
                    st.session_state.labels.pop()
                    if st.session_state.points:
                        # Repredict
                        mask = predict_mask(
                            st.session_state.image,
                            st.session_state.image_embedding,
                            st.session_state.points,
                            st.session_state.labels
                        )
                        st.session_state.current_mask = mask
                    else:
                        st.session_state.current_mask = None
                    st.rerun()

        with col_btn2:
            if st.button("Clear Current Selection"):
                st.session_state.points = []
                st.session_state.labels = []
                st.session_state.current_mask = None
                st.rerun()

    with col2:
        st.subheader("Segment Management")

        segment_name = st.text_input("Segment Name", placeholder="e.g., car_body")
        epsilon_factor = st.slider("Vectorization Simplification (epsilon)", min_value=0.001, max_value=0.05, value=0.005, step=0.001, format="%.3f")

        if st.button("Save Segment to SVG", disabled=st.session_state.current_mask is None or not segment_name):
            with st.spinner("Vectorizing..."):
                # 1. Convert mask to SVG path
                path_d = mask_to_svg_path(st.session_state.current_mask, epsilon_factor=epsilon_factor)

                # 2. Add to session state segments dictionary
                st.session_state.segments[segment_name] = path_d

                # 3. Inject the path into the working SVG string using lxml
                from lxml import etree

                try:
                    # Parse the original SVG string
                    root = etree.fromstring(st.session_state.original_svg.encode('utf-8'))

                    # Create the new group and path
                    group = etree.SubElement(root, f"{{{SVG_NS}}}g", id=segment_name)
                    path = etree.SubElement(
                        group,
                        f"{{{SVG_NS}}}path",
                        d=path_d,
                        fill="#FF0000",
                        opacity="0.5",
                        attrib={"fill-rule": "evenodd"} # Important for holes
                    )

                    # Update the stored original SVG with the newly injected element
                    st.session_state.original_svg = etree.tostring(root, pretty_print=True, encoding="unicode")

                    # Clear current selection for the next segment
                    st.session_state.points = []
                    st.session_state.labels = []
                    st.session_state.current_mask = None

                    st.success(f"Segment '{segment_name}' saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to inject SVG: {e}")

        # Display saved segments list
        if st.session_state.segments:
            st.write("### Saved Segments")
            for name in st.session_state.segments.keys():
                st.markdown(f"- **{name}**")

            # Provide download button for the final SVG
            st.download_button(
                label="Download Final SVG",
                data=st.session_state.original_svg,
                file_name="segmented_output.svg",
                mime="image/svg+xml"
            )
