import torch
from transformers import SamModel, SamProcessor
import streamlit as st
import numpy as np
from PIL import Image
from typing import Tuple, List


# Use @st.cache_resource to avoid reloading the model on every rerun
@st.cache_resource(show_spinner="Loading Segment Anything Model (SAM)...")
def load_sam_model() -> Tuple[SamModel, SamProcessor, str]:
    """Loads the SAM model and processor from Hugging Face."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using facebook/sam-vit-base as the standard baseline
    model_id = "facebook/sam-vit-base"
    model = SamModel.from_pretrained(model_id).to(device)
    processor = SamProcessor.from_pretrained(model_id)
    return model, processor, device


@st.cache_resource(show_spinner="Computing Image Embeddings...")
def compute_image_embedding(image: Image.Image) -> torch.Tensor:
    """
    Computes and caches the SAM image embedding for a given image.
    This is the heavy part of the computation.
    """
    model, processor, device = load_sam_model()

    # Preprocess the image to get pixel values
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Compute image embeddings
    with torch.no_grad():
        image_embeddings = model.get_image_embeddings(inputs.pixel_values)

    return image_embeddings


def predict_mask(
    image: Image.Image,
    image_embeddings: torch.Tensor,
    input_points: List[List[int]],
    input_labels: List[int],
) -> np.ndarray:
    """
    Predicts a binary mask given the image embeddings and prompt points.
    input_points: list of [x, y] coordinates
    input_labels: list of 1 (positive) or 0 (negative) for each point
    """
    model, processor, device = load_sam_model()

    # Format inputs for the processor
    # The processor expects points in the format [[[x1, y1], [x2, y2], ...]]
    # and labels in [[1, 0, ...]] for a single batch
    points = [input_points]
    labels = [input_labels]

    # Preprocess prompts
    inputs = processor(
        images=image, input_points=points, input_labels=labels, return_tensors="pt"
    ).to(device)

    # Run prediction using the cached embeddings
    with torch.no_grad():
        outputs = model(
            image_embeddings=image_embeddings,
            input_points=inputs.input_points,
            input_labels=inputs.input_labels,
            multimask_output=False,  # We only want the best mask
        )

    # Process the predicted mask back to the original image size
    # inputs contains original_sizes and reshaped_input_sizes from the processor call
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    # masks is a list of tensors, get the first one and squeeze it to a 2D array
    mask = masks[0]
    # Squeeze out the batch and channel dimensions if present, but keep spatial dims.
    # Usually shape is (1, 1, H, W) or (1, H, W)
    if mask.ndim > 2:
        mask = mask.squeeze()
        # If the image was 1x1, squeeze might have removed all dimensions.
        if mask.ndim < 2:
            mask = mask.view(masks[0].shape[-2], masks[0].shape[-1])

    mask = mask.numpy()

    # The mask is boolean, convert to uint8 for OpenCV (0 and 255)
    binary_mask = (mask * 255).astype(np.uint8)

    # To prevent "overlap onto a different part of the image", keep only the connected
    # components of the mask that actually contain at least one positive input point.
    import cv2
    num_labels, labels_img = cv2.connectedComponents(binary_mask)
    if num_labels > 1:
        # labels_img has values from 0 (background) to num_labels - 1
        filtered_mask = np.zeros_like(binary_mask)
        positive_points_labels = set()

        for pt, label in zip(input_points, input_labels):
            if label == 1:  # Positive point
                x, y = pt
                # Make sure point is within image bounds
                if 0 <= y < labels_img.shape[0] and 0 <= x < labels_img.shape[1]:
                    component_label = labels_img[y, x]
                    if component_label > 0:  # Ignore background
                        positive_points_labels.add(component_label)

        # Keep only the connected components that were clicked
        for comp_label in positive_points_labels:
            filtered_mask[labels_img == comp_label] = 255

        # If for some reason no positive points fell exactly on a mask component,
        # fallback to returning the original mask to prevent returning an empty mask.
        if len(positive_points_labels) > 0:
            binary_mask = filtered_mask

    return binary_mask
