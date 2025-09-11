# LiteMedSeg: Lightweight Medical Image Segmentation Models

LiteMedSeg is a collection of lightweight and efficient deep learning models for medical image segmentation. This repository provides implementations, training scripts, and evaluation tools for various state-of-the-art segmentation architectures, with a focus on medical datasets.

## Features

- Implementations of popular segmentation models:
  - UNet, UNet++, AttentionUNet, ResUNet, ResUNet++, DeepLabV3+, SCA-Inception-UNet, TransUNet, LiteMedSeg(ours)
- Ablation studies and model comparisons
- Support for multiple medical image datasets:
  - CVC-ClinicDB, Kvasir-SEG, Retina Blood Vessel
- Training, validation, and evaluation scripts
- Visualization and logging utilities
- Integration with Weights & Biases (wandb) for experiment tracking

## Repository Structure

```
.
├── ablation_study/                # Ablation study models, results, logs, visualizations
├── data/                          # Datasets (CVC-ClinicDB, Kvasir-SEG)
├── model_ablation_study/          # Ablation study training and utilities
├── model_comparisons/             # Models, evaluation results, logs, visualizations (created after running)
├── segmentation_models/           # Model implementations and utilities
├── wandb/                         # Weights & Biases logs
├── ablation_study.ipynb           # Ablation study notebook
├── [ModelName].ipynb              # Notebooks for each model
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- Recommended: Create a virtual environment

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/humblebeeintel/LiteMedSeg.git
   cd LiteMedSeg
   ```
2. Create an environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Environment variables:
   for using wandb for tracking, update `WANDB_API_KEY` in `.env.example` and run:
   ```bash
   cp .env.example .env
   ```

### Datasets

- Download and place datasets in the `data/` directory as follows:
  - `data/CVC-ClinicDB/`
  - `data/Kvasir-SEG/`

### Training & Evaluation

- Use the provided notebooks (e.g., `UNet.ipynb`, `DeepLabV3+.ipynb`) for model training and evaluation. Models, evaluation results and logs are saved in `model_comparisons/`.
- For ablation studies, refer to `ablation_study.ipynb` and the results will be saved to `model_ablation_study/` directory.

### Experiment Tracking

- This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking. Set up your wandb account and login before training.

