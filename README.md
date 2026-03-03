# SVG-SAMurai 🗡️

**SVG-SAMurai** is an interactive, Streamlit-based web application that leverages the power of Meta's **Segment Anything Model (SAM)** to transform raster (PNG, JPG) and vector (SVG) images into precisely segmented, editable SVG paths.

Whether you're starting from a flat image or an existing SVG file, SVG-SAMurai allows you to click on regions of interest, predict their boundaries, and inject those precise vector paths back into a master SVG document.

## 🌟 Features

- **Interactive Segmentation:** Click on any part of an uploaded image to instantly generate an accurate mask using the SAM Vision Transformer (`facebook/sam-vit-base`).
- **Support for Multiple Formats:** Upload PNG, JPG, or SVG files. Vector images are rasterized cleanly in the backend for processing, allowing you to segment them seamlessly.
- **Smart Vectorization:** Extracted masks are converted into optimized SVG `<path>` elements using the Ramer-Douglas-Peucker algorithm (via OpenCV) for smooth, simplified contours.
- **Adjustable Simplification:** Fine-tune the vectorization epsilon factor directly from the UI to control the complexity of the generated paths.
- **Segment Management:** Name your segments and save them into a live-updating SVG document.
- **In-Memory Caching:** Heavy image embeddings are cached securely using Streamlit's `@st.cache_data` and `@st.cache_resource`, ensuring snappy performance and instant mask prediction on subsequent clicks.
- **Easy Export:** Download the final composed SVG with all your tagged, labeled segments neatly organized in `<g>` groups.

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/), [streamlit-image-coordinates](https://pypi.org/project/streamlit-image-coordinates/)
* **Machine Learning:** [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) (Segment Anything Model)
* **Image Processing:** [OpenCV](https://opencv.org/) (Contour extraction & smoothing), [Pillow (PIL)](https://python-pillow.org/)
* **SVG / DOM Manipulation:** [lxml](https://lxml.de/) (XML parsing and injection), [CairoSVG](https://cairosvg.org/) (SVG rasterization)
* **Dependency Management:** [Poetry](https://python-poetry.org/)

## 🚀 Quick Start

### Prerequisites

- Python `>=3.10, <3.13` (Required for PyTorch and Triton compatibility)
- [Poetry](https://python-poetry.org/docs/#installation) installed on your system.
- System dependencies for CairoSVG and OpenCV (e.g., `libcairo2-dev`, `libgl1-mesa-glx` on Ubuntu/Debian).

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd svg-samurai
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

### Running the App

Start the Streamlit development server:

```bash
poetry run streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

## 📖 How to Use

1. **Upload an Image:** Use the file uploader to select a PNG, JPG, or SVG file. The app will calculate the complex image embeddings once (this may take a few moments depending on your hardware).
2. **Select Segments:** Click anywhere on the image in the left panel to prompt the model.
   - *Tip:* You can toggle the "Next Click is Negative Prompt" checkbox in the sidebar to exclude specific regions from your mask.
3. **Refine & Save:**
   - Use the "Undo Last Click" or "Clear Current Selection" buttons to fix mistakes.
   - Give your highlighted segment a descriptive name (e.g., `car_body`).
   - Adjust the **Simplification (epsilon)** slider if you want fewer, smoother nodes in your final vector path.
   - Click **Save Segment to SVG**.
4. **Download:** Once you have saved all desired segments, click the **Download Final SVG** button to retrieve your newly layered vector graphic.

## 📂 Project Structure

```text
svg-samurai/
├── app.py                  # Main Streamlit user interface and application state
├── pyproject.toml          # Poetry dependencies and project configuration
├── src/                    # Backend logic
│   ├── model.py            # PyTorch SAM loading, embedding generation, and mask prediction
│   ├── vectorizer.py       # OpenCV contour extraction and SVG path conversion
│   └── xml_manager.py      # lxml DOM manipulation and CairoSVG rasterization utilities
└── tests/                  # Unit tests for core logic
    ├── test_model.py
    ├── test_vectorizer.py
    └── test_xml_manager.py
```

## 🧪 Testing

The project uses `pytest` for unit testing. To run the test suite, simply execute:

```bash
poetry run pytest
```

---
*Developed with Streamlit and Meta's Segment Anything Model.*
