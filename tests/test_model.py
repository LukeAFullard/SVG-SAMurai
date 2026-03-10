import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import torch
from src.model import compute_image_embedding, predict_mask, load_sam_model


@pytest.fixture
def mock_sam_components():
    with patch("src.model.load_sam_model") as mock_load:
        # Create mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_device = "cpu"

        # Configure mock model's get_image_embeddings
        mock_embeddings = torch.zeros((1, 256, 64, 64))
        mock_model.get_image_embeddings.return_value = mock_embeddings

        # Configure mock model's forward pass
        mock_outputs = MagicMock()
        # Shape matches outputs.pred_masks in reality (batch, num_masks, height, width)
        mock_outputs.pred_masks = torch.ones((1, 1, 100, 100))
        mock_model.return_value = mock_outputs

        # Configure processor
        mock_processor_output = MagicMock()
        mock_processor_output.pixel_values = torch.zeros((1, 3, 100, 100))
        mock_processor_output.input_points = torch.zeros((1, 1, 2))
        mock_processor_output.input_labels = torch.zeros((1, 1))
        # Provide dictionary access to mimic processor output dictionary
        mock_processor_output.__getitem__.side_effect = lambda key: {
            "original_sizes": torch.tensor([[100, 100]]),
            "reshaped_input_sizes": torch.tensor([[100, 100]]),
        }[key]

        # Allow processor to act like a callable and return the configured output
        mock_processor.return_value = mock_processor_output
        mock_processor_output.to.return_value = mock_processor_output

        # Mock post_process_masks
        mock_post_process = MagicMock()
        # post_process_masks returns a list of tensors
        # Use a real tensor to avoid mock chain confusion with .numpy() and type checking
        mock_mask_tensor = torch.ones((1, 1, 100, 100), dtype=torch.float32)
        mock_post_process.return_value = [mock_mask_tensor]
        mock_processor.image_processor.post_process_masks = mock_post_process

        # Set the mocked load function to return these objects
        mock_load.return_value = (mock_model, mock_processor, mock_device)

        yield mock_load, mock_model, mock_processor


def test_load_sam_model_cuda_available():
    """Test load_sam_model initializes with CUDA when available."""
    # Clear streamlit cache before testing
    load_sam_model.clear()

    with patch("src.model.torch.cuda.is_available", return_value=True), \
         patch("src.model.SamModel.from_pretrained") as mock_model_from_pretrained, \
         patch("src.model.SamProcessor.from_pretrained") as mock_processor_from_pretrained:

        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model

        model, processor, device = load_sam_model()

        mock_model_from_pretrained.assert_called_once_with("facebook/sam-vit-base")
        mock_model.to.assert_called_once_with("cuda")
        mock_processor_from_pretrained.assert_called_once_with("facebook/sam-vit-base")
        assert device == "cuda"


def test_load_sam_model_cpu_available():
    """Test load_sam_model initializes with CPU when CUDA is not available."""
    # Clear streamlit cache before testing
    load_sam_model.clear()

    with patch("src.model.torch.cuda.is_available", return_value=False), \
         patch("src.model.SamModel.from_pretrained") as mock_model_from_pretrained, \
         patch("src.model.SamProcessor.from_pretrained") as mock_processor_from_pretrained:

        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model

        model, processor, device = load_sam_model()

        mock_model_from_pretrained.assert_called_once_with("facebook/sam-vit-base")
        mock_model.to.assert_called_once_with("cpu")
        mock_processor_from_pretrained.assert_called_once_with("facebook/sam-vit-base")
        assert device == "cpu"


def test_compute_image_embedding(mock_sam_components):
    mock_load, mock_model, mock_processor = mock_sam_components

    # Create a dummy image
    image = Image.new("RGB", (100, 100), color="white")

    # Run the function
    embeddings = compute_image_embedding(image)

    # Assertions
    mock_load.assert_called_once()
    mock_processor.assert_called_once()
    mock_model.get_image_embeddings.assert_called_once()

    # Check that the return value matches what we mocked
    assert embeddings.shape == (1, 256, 64, 64)


def test_predict_mask(mock_sam_components):
    mock_load, mock_model, mock_processor = mock_sam_components

    # Dummy inputs
    image = Image.new("RGB", (100, 100), color="white")
    image_embeddings = torch.zeros((1, 256, 64, 64))
    input_points = [[50, 50]]
    input_labels = [1]

    # Run prediction
    mask = predict_mask(image, image_embeddings, input_points, input_labels)

    # Assertions
    mock_load.assert_called_once()

    # The processor should be called with input_points and input_labels
    args, kwargs = mock_processor.call_args
    assert "input_points" in kwargs
    assert kwargs["input_points"] == [input_points]
    assert kwargs["input_labels"] == [input_labels]

    # The model should be called to make the prediction
    mock_model.assert_called_once()

    # The mask should be a numpy array of uint8
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8
    assert mask.shape == (100, 100)  # Since squeeze() removes 1-dims

    # Since we mocked the output to be all ones, the scaled mask should be all 255s
    assert np.all(mask == 255)


def test_predict_mask_negative_label(mock_sam_components):
    """Test that predict_mask handles negative labels (0) correctly."""
    mock_load, mock_model, mock_processor = mock_sam_components

    # Dummy inputs
    image = Image.new("RGB", (100, 100), color="white")
    image_embeddings = torch.zeros((1, 256, 64, 64))
    input_points = [[50, 50]]
    input_labels = [0]  # Negative label

    # Run prediction
    mask = predict_mask(image, image_embeddings, input_points, input_labels)

    # Assertions
    mock_load.assert_called_once()

    # The processor should be called with input_points and input_labels
    _, kwargs = mock_processor.call_args
    assert "input_points" in kwargs
    assert kwargs["input_points"] == [input_points]
    assert kwargs["input_labels"] == [input_labels]

    # The mask should be a numpy array of uint8
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8
    assert mask.shape == (100, 100)

    # Even with a negative point, it should return the mask if it's the only point
    # provided (though usually you'd have at least one positive point).
    # Based on the implementation, if no positive points are found, it returns the raw mask.
    assert np.all(mask == 255)
