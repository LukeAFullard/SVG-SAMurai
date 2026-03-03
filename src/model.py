import torch
from transformers import SamModel, SamProcessor
import streamlit as st
import numpy as np

# Use @st.cache_resource to avoid reloading the model on every rerun
@st.cache_resource(show_spinner="Loading Segment Anything Model (SAM)...")
def load_sam_model():
    """Loads the SAM model and processor from Hugging Face."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using facebook/sam-vit-base as the standard baseline
    model_id = "facebook/sam-vit-base"
    model = SamModel.from_pretrained(model_id).to(device)
    processor = SamProcessor.from_pretrained(model_id)
    return model, processor, device


@st.cache_data(show_spinner="Computing Image Embeddings...")
def compute_image_embedding(image):
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


def predict_mask(image, image_embeddings, input_points, input_labels):
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
        images=image,
        input_points=points,
        input_labels=labels,
        return_tensors="pt"
    ).to(device)

    # Run prediction using the cached embeddings
    with torch.no_grad():
        outputs = model(
            image_embeddings=image_embeddings,
            input_points=inputs.input_points,
            input_labels=inputs.input_labels,
            multimask_output=False, # We only want the best mask
        )

    # outputs.pred_masks is (batch_size, num_masks, height, width)
    # Get the predicted mask and squeeze it to a 2D array
    mask = outputs.pred_masks.squeeze().cpu().numpy()

    # The mask is boolean, convert to uint8 for OpenCV (0 and 255)
    binary_mask = (mask * 255).astype(np.uint8)

    return binary_mask
