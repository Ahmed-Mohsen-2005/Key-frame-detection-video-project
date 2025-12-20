# Key-frame-detection-video-project
## Overview

This project implements two distinct Deep Learning approaches for automatic video summarization: a **Supervised Bi-LSTM with Attention** and an **Unsupervised Autoencoder with Clustering**. The goal is to condense long videos into a short summary by identifying and extracting the most significant "key frames" that capture the essence of the video content.

The notebook demonstrates these methods using the **SumMe** and **TVSum** benchmark datasets.

## Key Features

* **Hybrid Framework:** Utilizes both **PyTorch** (for the Bi-LSTM/ResNet pipeline) and **TensorFlow/Keras** (for the Autoencoder pipeline).
* **Feature Extraction:** Leverages a pre-trained **ResNet50** model to extract high-level visual features from video frames.
* **Dual Approaches:**
1. **Supervised Approach:** Bidirectional LSTM with an Attention mechanism to score frame importance.
2. **Unsupervised Approach:** Convolutional Autoencoder to learn efficient frame representations, followed by K-Means clustering to select diverse key frames.


* **Visualization:** Automatically generates and saves grid visualizations of the top selected key frames.

## Dataset

The project is configured to use the **SumMe** and **TVSum** video summarization datasets.

* **Source:** Downloaded automatically via `kagglehub`.
* **Dataset Path:** `/kaggle/input/summe-videos-and-tvsum-videos`

## Prerequisites

To run this notebook, you need the following Python libraries installed:

* Python 3.x
* `kagglehub`
* `opencv-python`
* `numpy`
* `pandas`
* `torch` & `torchvision`
* `tensorflow`
* `scikit-learn`
* `matplotlib`
* `Pillow`

You can install the dependencies using pip:

```bash
pip install opencv-python numpy pandas torch torchvision scikit-learn matplotlib tensorflow kagglehub

```

## Model Architectures

### 1. Bi-LSTM Summarizer (Supervised)

* **Framework:** PyTorch
* **Input:** 2048-dimensional feature vectors extracted from video frames using a pre-trained ResNet50.
* **Structure:**
* **Bidirectional LSTM:** Captures temporal dependencies in both forward and backward directions (2 layers, 256 hidden dim).
* **Attention Layer:** Computes attention weights to focus on salient parts of the video sequence.
* **Regressor:** A linear layer followed by a Sigmoid activation to output an importance score (0-1) for each frame.



### 2. Autoencoder Summarizer (Unsupervised)

* **Framework:** TensorFlow / Keras
* **Input:** Raw RGB video frames resized to (224, 224, 3).
* **Structure:**
* **Encoder:** A series of Conv2D and MaxPooling2D layers to compress frames into a compact latent representation.
* **Decoder:** Upsampling and Conv2D layers to reconstruct the original frame from the latent space.


* **Key Frame Selection:**
* The Autoencoder is trained on the specific target video to learn its visual distribution.
* **K-Means Clustering** is applied to the learned latent representations to group similar frames.
* The frame closest to each cluster center is selected as a key frame, ensuring a diverse summary.



## Configuration

You can adjust the following parameters in the `SETUP & CONFIGURATION` section of the notebook:

| Parameter | Default | Description |
| --- | --- | --- |
| `TARGET_VIDEO_PATH` | *path to a specific video* | The absolute path to the input video file. |
| `OUTPUT_DIR` | `./specific_video_result` | Directory where results and plots will be saved. |
| `SKIP_FRAMES` | `15` | Frame sampling rate (e.g., process 1 frame every 15 frames). |
| `TOP_K_FRAMES` | `25` | The number of key frames to include in the final summary. |

## Usage

1. **Download Data:** The first cell automatically downloads the dataset using `kagglehub`.
2. **Run Pipeline:** Execute the main block. The script will:
* Load the target video.
* Extract frames based on `SKIP_FRAMES`.
* **Run Model 1:** Extract ResNet features, run the Bi-LSTM (demo inference), and plot the top scored frames.
* **Run Model 2:** Train the Autoencoder on the video frames, cluster the latent space, and plot the representative frames.


3. **View Results:** Check the `specific_video_result` folder for the generated `.png` visualizations showing the selected key frames for both models.

## Outputs

The notebook generates visualization images containing a grid of the selected key frames.

* **Filename Format:** `{video_name}_{Model_Name}.png`
* Example: `-esJrBWj2d8_Model1_BiLSTM.png`
* Example: `-esJrBWj2d8_Model2_Autoencoder.png`


