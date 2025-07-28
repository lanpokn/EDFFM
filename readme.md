# Feature-Augmented Mamba for Real-time Event Denoising (EventMamba-FX)

This repository contains the official implementation for the research project "Feature-Augmented Mamba for Real-time Event Denoising and Artifact Removal". The project aims to develop a lightweight and efficient deep learning framework for real-time, event-by-event removal of noise (BA) and artifacts (Flicker, Flare) from event camera streams.

Our core idea is to combine the physical insights of classical filters (via feature engineering) with the long-range sequence modeling capabilities of Mamba, creating a "Feature-Augmented" pure sequence Mamba model that is both powerful and efficient.

## 1. Environment Setup (Windows with NVIDIA 4060)

This project is developed using PyTorch and the official Mamba library. We strongly recommend using Conda for environment management.

Step 1: Create and Activate Conda Environment

Open your Anaconda Prompt or terminal and run the following commands:

Generated bash
# Create a new conda environment with Python 3.10
conda create -n event-mamba-fx python=3.10

# Activate the newly created environment
conda activate event-mamba-fx


Step 2: Install PyTorch with CUDA Support

Your NVIDIA GeForce RTX 4060 supports CUDA 12. Go to the official PyTorch website to get the latest stable installation command. It should look similar to this:

Generated bash
# Command for PyTorch 2.3 with CUDA 12.1 (Please verify on the official website)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

Step 3: Install Mamba and its Dependencies

Mamba requires specific dependencies for its optimized kernels.

Generated bash
# Install build dependencies
pip install ninja

# Install a key dependency for Mamba
pip install causal-conv1d>=1.2.0

# Install the official Mamba package
pip install mamba-ssm

Step 4: Install Other Necessary Libraries

Generated bash
# For data handling, configuration, progress bars, and evaluation
pip install numpy pyyaml tqdm scikit-learn pandas opencv-python h5py

After these steps, your environment is ready for development and training.

## 2. Project Structure

The project is organized into several directories to separate concerns like data simulation, model architecture, training logic, and evaluation.

```
event-mamba-fx/
│
├── data/                       # Directory for storing datasets
│   ├── raw_videos/             # Raw video files (e.g., from KITTI, TartanAir)
│   └── simulated_events/       # Generated paired event data
│
├── simulator/                  # Scripts for data generation
│
├── src/                        # Main source code
│   ├── datasets.py             # PyTorch Dataset and DataLoader classes
│   ├── feature_extractor.py    # The core Feature-Augmented module
│   ├── model.py                # Definition of the Mamba-based denoising network
│   ├── trainer.py              # The main training and validation loop logic
│   └── evaluate.py             # Script for evaluating models on the test set
│
├── configs/                    # Configuration files for experiments
│   ├── base_config.yaml        # Base configuration with default parameters
│   └── experiment_01.yaml      # Specific config for a particular experiment
│
├── checkpoints/                # To save trained model weights
│
├── results/                    # To save evaluation results, logs, and plots
│
├── notebooks/                  # Jupyter notebooks for visualization and analysis
│   └── 01_data_visualization.ipynb
│   └── 02_results_analysis.ipynb
│
├── main.py                     # Main entry point to run training and evaluation
├── requirements.txt            # List of python package dependencies
└── README.md                   # This file
```

## 3. How to Run (Future Steps)
