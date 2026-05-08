# TLIO (Tight-Learned Inertial Odometry) - Workspace Overview

This document provides a directory structure and a brief description of the files in the workspace.

## Root Directory

- `LICENSE`: Project license.
- `README.md`: Main documentation with setup, training, testing, and evaluation instructions.
- `environment.yaml`: Conda environment configuration file for setting up the project dependencies.
- `requirements.txt`: Standard Python requirements file.
- `docs/`: Jekyll-based documentation site.
- `tlio_golden/`: Dataset directory containing IMU sequences, ground truth data, and list files (`train_list.txt`, `val_list.txt`, `test_list.txt`).
- `src/`: Core source code directory.

---

## Source Code (`src/`)

### Main Scripts
- `main_net.py`: entry point for training, testing, and evaluating the neural network for displacement and covariance estimation.
- `main_filter.py`: entry point for running the Stochastic Cloning Extended Kalman Filter (EKF) using network measurements.
- `convert_model_to_torchscript.py`: converts trained PyTorch models to TorchScript format for use in the EKF tracker.
- `plot_filter_state.py`: script for visualizing filter states, trajectories, and comparing against ground truth.

### Subpackages

#### `network/` - Neural Network Modules
- `model_factory.py`: Handles creation of different network architectures.
- `model_resnet.py` & `model_resnet_seq.py`: Implementation of ResNet-based architectures for IMU processing.
- `model_tcn.py`: Implementation of Temporal Convolutional Network (TCN) architectures.
- `train.py`: contains the training loop and optimization logic.
- `test.py` & `eval.py`: scripts for testing models and computing performance metrics.
- `losses.py`: custom loss functions for displacement and covariance estimation.
- `covariance_parametrization.py`: utilities for parameterizing and estimating uncertainty.

#### `tracker/` - EKF and State Estimation
- `imu_tracker.py`: Core logic for IMU-based state estimation.
- `scekf.py`: Implementation of the Stochastic Cloning Extended Kalman Filter.
- `imu_tracker_runner.py`: Wrapper to run the tracker on datasets.
- `imu_buffer.py`: Manages buffers for IMU measurements.
- `imu_calib.py`: Utilities for IMU calibration and bias handling.
- `meas_source_torchscript.py`: Interface to use TorchScript models as measurement sources in the filter.

#### `dataloader/` - Data Loading and Preprocessing
- `tlio_data.py`: Main interface for accessing TLIO datasets.
- `sequences_dataset.py`: standard PyTorch dataset for IMU sequences.
- `memmapped_sequences_dataset.py`: memory-mapped dataset for efficient loading of large datasets.
- `iterable_pseudorandom_sequences_dataset.py`: iterable-style dataset for very large datasets.
- `data_io.py`: handlers for reading and writing data files.
- `data_transform.py`: implements data augmentations and coordinate transformations.
- `constants.py`: shared constants for data dimensions and frequencies.

#### `utils/` - Shared Utilities
- `math_utils.py` & `torch_math_utils.py`: general and PyTorch-specific mathematical functions (rotations, etc.).
- `alignment.py`: tools for aligning estimated trajectories with ground truth (e.g., Umeyama algorithm).
- `logging.py`: central logging configuration.
- `argparse_utils.py`: helpers for consistent CLI argument parsing.
- `o3d_visualizer.py`: interactive 3D visualization using Open3D.

#### `analysis/` - Evaluation and Visualization
- `display_json.py`: tool to analyze `metrics.json` files and plot statistics using Pandas/Matplotlib.
- `plot_comparison_between_run.py`: utility to compare trajectories and errors across different experiments.

#### `batch_runner/` - Batch Processing Scripts
- `net_test_batch.py`: runs network testing across multiple models and datasets.
- `filter_batch.py`: runs the EKF across multiple configurations in batch.
- `plot_batch.py`: automates plot generation for batch runs.

#### `doc/`
- `Readme_Filter_Evaluation_pipline.md`: Documentation specifically for the filter evaluation pipeline.
