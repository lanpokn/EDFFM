# Feature-Augmented Mamba for Real-time Event Denoising (EventMamba-FX)

This repository contains the official implementation for the research project "Feature-Augmented Mamba for Real-time Event Denoising and Artifact Removal". The project aims to develop a lightweight and efficient deep learning framework for real-time, event-by-event removal of noise (BA) and artifacts (Flicker, Flare) from event camera streams.

Our core idea is to combine the physical insights of classical filters (via feature engineering) with the long-range sequence modeling capabilities of Mamba, creating a "Feature-Augmented" pure sequence Mamba model that is both powerful and efficient.

## 1. Environment Setup (IMPORTANT - Use Existing Environment)

⚠️ **CRITICAL**: This project MUST use the existing `event_flare` conda environment. Do NOT create a new environment or install additional packages to avoid dependency conflicts.

### Using the Existing Environment

```bash
# Activate the existing environment
conda activate event_flare

# Verify the environment is activated
conda info --envs
```

The `event_flare` environment already contains all necessary dependencies including:
- PyTorch with CUDA support
- Mamba SSM library 
- All required Python packages (numpy, yaml, tqdm, etc.)

### If Environment Setup is Required (Emergency Only)

Only use this if the `event_flare` environment is missing or corrupted:

```bash
# Create environment with Python 3.10
conda create -n event_flare python=3.10

# Activate environment
conda activate event_flare

# Install PyTorch with CUDA 12.1 support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Mamba dependencies
pip install ninja causal-conv1d>=1.2.0 mamba-ssm

# Install other dependencies
pip install numpy pyyaml tqdm scikit-learn pandas opencv-python h5py
```

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

## 3. Key Features

### PFD (Polarity-Focused Denoising) Integration

This implementation includes advanced PFD features based on academic research for enhanced event denoising:

- **Mp (Polarity Map)**: Tracks latest polarity at each pixel
- **Mf (Polarity Frequency)**: Counts polarity changes within time windows  
- **Ma (Neighborhood Activity)**: Aggregates polarity changes in spatial neighborhoods
- **D(x,y) (Density Score)**: Computes polarity change density for noise detection

### Configurable Neighborhood Sizes

The PFD features support configurable neighborhood sizes through the config file:

```yaml
feature_extractor:
  pfd_neighborhood_size: 3    # Options: 1 (1x1), 3 (3x3), 5 (5x5)
  pfd_time_window: 25000      # Time window for Mf calculation (25ms in µs)
```

### Resolution-Independent Features

- **Center-relative coordinates**: Spatial features normalized to [-1,1] range
- **Log-scaled temporal features**: Numerical stability for varying time scales
- **Generalization**: Model works across different event camera resolutions

## 4. Quick Start

### Testing Feature Extraction

Test the PFD feature extraction with different neighborhood sizes:

```bash
# Activate environment
conda activate event_flare

# Run feature extraction test
python test_features.py
```

This will test both 1x1 and 3x3 neighborhood modes and display feature analysis.

### Training the Model

```bash
# Activate environment
conda activate event_flare

# Run training
python main.py
```

### Configuration

Edit `configs/config.yaml` to adjust:

- **Model architecture**: `d_model`, `n_layers`, `d_state`
- **PFD parameters**: `pfd_neighborhood_size`, `pfd_time_window`  
- **Training settings**: `batch_size`, `learning_rate`, `epochs`
- **Data paths**: Update paths to your dataset

## 5. PFD Feature Details

The feature vector contains 32 dimensions including:

| Index | Feature | Description |
|-------|---------|-------------|
| 0-1 | x_center, y_center | Center-relative spatial coordinates [-1,1] |
| 2 | polarity | Event polarity {-1, 1} |
| 3 | dt_norm | Log-scaled delta time between events |
| 4 | dt_pixel_norm | Log-scaled time since last event at pixel |
| 5 | mf_current | Polarity frequency (Mf) in time window |
| 6 | ma_neighborhood | Neighborhood polarity changes (Ma) |
| 7 | ne_neighborhood | Active neighbor count (Ne) |
| 8 | d_neighborhood | Polarity change density D(x,y) = Ma/Ne |
| 9 | pfd_a_score | PFD-A denoising score |
| 10 | pfd_b_score | PFD-B flicker detection score |
| 11 | polarity_changes | Total polarity changes at pixel |
| 12 | event_count | Total events at pixel |
| 13-31 | (reserved) | Additional feature expansion |

## 6. Architecture Overview

The model combines:

1. **Feature Extractor** (`src/feature_extractor.py`): Converts raw events to PFD-enhanced features
2. **Mamba Backbone** (`src/model.py`): Sequence modeling with state-space models
3. **Training Pipeline** (`src/trainer.py`): End-to-end training with validation
4. **Evaluation** (`src/evaluate.py`): Performance metrics and analysis

## 7. Important Files

### Core Implementation Files

- **`src/feature_extractor.py`**: PFD feature extraction with parameterized neighborhoods
- **`configs/config.yaml`**: Configuration file with PFD parameters
- **`test_features.py`**: Testing script for feature extraction validation
- **`main.py`**: Training entry point

### Key Configuration Parameters

```yaml
# PFD Feature Settings
feature_extractor:
  pfd_time_window: 25000      # Time window for Mf calculation (µs)
  pfd_neighborhood_size: 3    # Neighborhood size (1, 3, or 5)

# Model Architecture
model:
  input_feature_dim: 32       # Feature vector dimension
  d_model: 128               # Mamba model dimension
  n_layers: 4                # Number of Mamba layers
```

## 8. Implementation Notes

### Neighborhood Size Selection

- **1x1**: Fastest processing, pixel-only features
- **3x3**: Balanced performance/accuracy (recommended)
- **5x5**: Maximum context, higher computational cost

### Memory and Performance

- Feature extraction scales O(N) with sequence length
- Memory usage depends on resolution and sequence length
- PFD maps maintain state across event sequence

### Generalization Improvements

- Removed absolute timestamps (replaced with delta time)
- Center-relative spatial coordinates for resolution independence
- Log-scaled features for numerical stability across time ranges

## 9. Testing and Validation

Run the comprehensive test suite:

```bash
conda activate event_flare
python test_features.py
```

Expected output includes:
- Feature extraction for both 1x1 and 3x3 modes
- Spatial coordinate range verification [-1,1]
- PFD feature value analysis
- Polarity change detection validation

## 10. Troubleshooting

### Environment Issues
- Always use `conda activate event_flare`
- Do not install additional packages
- Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

### Configuration Errors
- Ensure `pfd_neighborhood_size` is odd number (1, 3, 5)
- Check `pfd_time_window` is positive integer
- Verify data paths in config exist

### Feature Extraction Issues
- Check input event format: [x, y, t, p]
- Ensure events are time-ordered
- Verify resolution matches config settings
