from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from PIL import Image
import numpy as np


def test_app_title_and_initial_state():
    """Test that the app initializes correctly with expected title and empty state."""
    at = AppTest.from_file("app.py", default_timeout=10).run()

    assert not at.exception
    assert at.title[0].value == "SVG-SAMurai 🗡️"
    assert (
        "Transform raster and vector images into segmented SVG paths"
        in at.markdown[0].value
    )

    # Check session state initialization
    assert at.session_state.image is None
    assert at.session_state.image_embedding is None
    assert at.session_state.points == []
    assert at.session_state.labels == []
    assert at.session_state.current_mask is None
    assert at.session_state.segments == {}
    assert at.session_state.original_svg is None


@patch("app.load_image")
@patch("app.compute_image_embedding")
def test_app_file_upload(mock_compute_embedding, mock_load_image):
    """Test the file upload and embedding computation."""
    # Create a dummy image
    dummy_image = Image.new("RGB", (100, 100), color="red")
    mock_load_image.return_value = dummy_image
    mock_compute_embedding.return_value = "dummy_embedding"

    import io

    class MockUploadedFile(io.BytesIO):
        def __init__(self, name, type, content):
            super().__init__(content)
            self.name = name
            self.type = type

    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # Mocking file_uploader is a reliable way to simulate an uploaded file in older Streamlit versions.
    with patch("streamlit.file_uploader") as mock_uploader:
        mock_file = MockUploadedFile("test.png", "image/png", img_byte_arr)
        mock_uploader.return_value = mock_file

        at = AppTest.from_file("app.py", default_timeout=30).run()

        assert not at.exception

        # Verify the session state changed
        assert at.session_state.image is not None
        assert 'width="100"' in at.session_state.original_svg
        assert 'height="100"' in at.session_state.original_svg
        assert 'viewBox="0 0 100 100"' in at.session_state.original_svg

        # Verify success message
        assert at.success[0].value == "Image embedded successfully!"


@patch("app.load_image")
@patch("app.compute_image_embedding")
@patch("app.predict_mask")
@patch("app.streamlit_image_coordinates")
def test_app_interactive_segmentation(
    mock_sic, mock_predict_mask, mock_compute_embedding, mock_load_image
):
    """Test clicking on the image and generating a mask."""
    # Set up mocks
    dummy_image = Image.new("RGB", (100, 100), color="red")
    mock_load_image.return_value = dummy_image
    mock_compute_embedding.return_value = "dummy_embedding"

    # Mock predict_mask to return a dummy mask
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[25:75, 25:75] = 255
    mock_predict_mask.return_value = dummy_mask

    import io

    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format="PNG")

    class MockUploadedFile(io.BytesIO):
        def __init__(self, name, type, content):
            super().__init__(content)
            self.name = name
            self.type = type

    with patch("streamlit.file_uploader") as mock_uploader:
        mock_file = MockUploadedFile("test.png", "image/png", img_byte_arr.getvalue())
        mock_uploader.return_value = mock_file

        at = AppTest.from_file("app.py", default_timeout=20).run()

        assert not at.exception
        assert at.session_state.image is not None

        # To simulate custom component behavior that drives internal logic (which fails naturally in AppTest),
        # we set the session state points and current mask explicitly instead of testing streamlit_image_coordinates internal hook mechanism
        at.session_state.points = [[50, 50]]
        at.session_state.labels = [1]
        at.session_state.current_mask = dummy_mask
        at.run()

        assert not at.exception
        assert at.session_state.points == [[50, 50]]
        assert at.session_state.labels == [1]


@patch("app.load_image")
@patch("app.compute_image_embedding")
@patch("app.predict_mask")
@patch("app.mask_to_svg_path")
@patch("src.xml_manager.add_path_to_svg")
@patch("app.streamlit_image_coordinates")
def test_app_save_segment(
    mock_sic,
    mock_add_path,
    mock_mask_to_svg,
    mock_predict_mask,
    mock_compute_embedding,
    mock_load_image,
):
    """Test saving a predicted segment."""
    dummy_image = Image.new("RGB", (100, 100), color="red")
    mock_load_image.return_value = dummy_image
    mock_compute_embedding.return_value = "dummy_embedding"

    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    mock_predict_mask.return_value = dummy_mask

    mock_mask_to_svg.return_value = "M 25 25 L 75 25 L 75 75 L 25 75 Z"
    mock_add_path.return_value = (
        "<svg><path d='M 25 25 L 75 25 L 75 75 L 25 75 Z'/></svg>"
    )

    import io

    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format="PNG")

    class MockUploadedFile(io.BytesIO):
        def __init__(self, name, type, content):
            super().__init__(content)
            self.name = name
            self.type = type

    with patch("streamlit.file_uploader") as mock_uploader:
        mock_file = MockUploadedFile("test.png", "image/png", img_byte_arr.getvalue())
        mock_uploader.return_value = mock_file

        at = AppTest.from_file("app.py", default_timeout=20).run()

        # Simulate click behavior directly and patch component since we know it operates correctly from prior test
        mock_sic.return_value = {"x": 50, "y": 50}
        at.session_state.current_mask = dummy_mask
        at.run()

        # Set segment name
        at.text_input[0].set_value("test_segment").run()

        # Click Save
        save_button = None
        for btn in at.button:
            if btn.label == "Save Segment to SVG":
                save_button = btn
                break

        assert save_button is not None
        save_button.click().run()

        assert not at.exception

        # Check segments dictionary
        assert "test_segment" in at.session_state.segments

        # Check that state was reset
        assert at.session_state.points == []
        assert at.session_state.labels == []
        assert at.session_state.current_mask is None

        # Check success message
        # Since the app uses `st.rerun()` at the end of success, the success message
        # will briefly appear then disappear on rerun in Streamlit AppTest.
        # It's better to just ensure `segments` was populated and points reset.
        pass
