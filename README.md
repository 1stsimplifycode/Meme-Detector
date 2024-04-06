---

# Meme Classification Project

This project utilizes a pre-trained ResNet-18 model within the PyTorch framework to classify memes into "harmful" or "not harmful" categories.

## Installation

Ensure Python 3.6+ is installed on your system. You can download the latest version of Python from [python.org](https://www.python.org/).

After installing Python, use pip, the Python package installer, to install the necessary libraries. Open your command line or terminal and run:

```bash
pip install torch torchvision pandas pillow
```

This will install PyTorch, torchvision for image processing utilities, pandas for CSV file handling, and Pillow for image manipulation.

## How to Install the Project

1. **Clone or Download the Repository**: If you have git installed, you can clone the repository using the following command:

    ```bash
    git clone [https://github.com/1stsimplifycode/Meme-Detector.git]
    ```

    Alternatively, you can download the repository as a ZIP file from the GitHub page and extract it.

2. **Prepare Your Dataset**: Ensure your dataset is structured appropriately, with a directory for images and a CSV file listing the image filenames and their labels. The expected structure is:

    - Images directory: `C:\HATEFUL ANALYSIS\hateful_memes`
    - CSV file: `C:\HATEFUL ANALYSIS\hateful_memes\hateful_memes_original.csv`

3. **Navigate to the Project Directory**: Use the command line or terminal to navigate to the directory where you cloned or extracted the project.

    ```bash
    cd path\to\meme_classification_project
    ```

4. **Run the Script**: Execute the Python script with the following command:

    ```bash
    python multimodal_hateful_meme_detection.py
    ```

    Follow any on-screen prompts to input the dataset name or CSV file path if required by the script.

## Usage

After installation, you can run the script as detailed above to classify memes. The script will automatically train the model on your dataset and output the training progress and the accuracy on the test set.

---
