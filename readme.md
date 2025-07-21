# Feature-Augmented Mamba for Real-time Event Denoising (EventMamba-FX)

This repository contains the official implementation for the research project "Feature-Augmented Mamba for Real-time Event Denoising and Artifact Removal". The project aims to develop a lightweight and efficient deep learning framework for real-time, event-by-event removal of noise (BA) and artifacts (Flicker, Flare) from event camera streams.

Our core idea is to combine the physical insights of classical filters (via feature engineering) with the long-range sequence modeling capabilities of Mamba, creating a "Feature-Augmented" pure sequence Mamba model that is both powerful and efficient.

## 1. Environment Setup (Windows with NVIDIA 4060)

This project is developed using PyTorch and the official Mamba library. We strongly recommend using Conda for environment management.

**Step 1: Create and Activate Conda Environment**

Open your Anaconda Prompt or terminal and run the following commands:

```bash
# Create a new conda environment with Python 3.10
conda create -n event-mamba-fx python=3.10

# Activate the newly created environment
conda activate event-mamba-fx
```

**Step 2: Install PyTorch with CUDA Support**

Your NVIDIA GeForce RTX 4060 supports CUDA 12. Go to the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the latest stable installation command. It should look similar to this:

```bash
# Command for PyTorch 2.3 with CUDA 12.1 (Please verify on the official website)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 3: Install Mamba and its Dependencies**

Mamba requires specific dependencies for its optimized kernels.

```bash
# Install build dependencies
pip install ninja

# Install a key dependency for Mamba
pip install causal-conv1d>=1.2.0

# Install the official Mamba package
pip install mamba-ssm
```

**Step 4: Install Other Necessary Libraries**

```bash
# For data handling, progress bars, etc.
pip install numpy tqdm opencv-python h5py pandas```

After these steps, your environment is ready for development and training.

## 2. Project Structure

The project is organized into several directories to separate concerns like data simulation, model architecture, training logic, and evaluation.

```
event-mamba-fx/
│
├── data/                       # Directory for storing datasets
│   ├── raw_videos/             # Raw video files (e.g., from KITTI, TartanAir)
│   └── simulated_events/       # Generated paired (noisy, clean_labels) event data
│
├── simulator/                  # Scripts for data generation
│   ├── v2e.py                  # Core V2E simulator logic (can be a submodule)
│   ├── noise_models.py         # Your implementation of BA, Flicker, Flare noise
│   └── generate_dataset.py     # Main script to run simulation and create paired data
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

**(This section is a template for you to fill in as you complete the project)**

**Step 1: Generate the Dataset**

```bash
python simulator/generate_dataset.py --config configs/dataset_gen_config.yaml
```

**Step 2: Train the Model**

```bash
python main.py --mode train --config configs/experiment_01.yaml
```

**Step 3: Evaluate the Model**

```bash
python main.py --mode evaluate --checkpoint checkpoints/best_model.pth --config configs/experiment_01.yaml
```

## 4. Work Plan & To-Do

-   [ ] **Phase 1: Infrastructure**
    -   [ ] Implement noise models in `simulator/noise_models.py`.
    -   [ ] Finalize data generation script `simulator/generate_dataset.py`.
    -   [ ] Implement PFD baseline and evaluation metrics in `src/evaluate.py`.
-   [ ] **Phase 2: Prototyping**
    -   [ ] Implement the streaming feature extractor in `src/feature_extractor.py`.
    -   [ ] Define the Mamba network in `src/model.py`.
    -   [ ] Implement the training loop in `src/trainer.py` and get the model to converge.
-   [ ] **Phase 3: Iteration & Experiments**
    -   [ ] Conduct feature ablation studies.
    -   [ ] Tune Mamba hyperparameters.
    -   [ ] Run final comparisons against baselines.
-   [ ] **Phase 4: Publication**
    -   [ ] Write the paper.
    -   [ ] Clean up and release the code.