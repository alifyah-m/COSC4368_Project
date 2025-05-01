# COVID-19 X-Ray Classification

This repository contains the implementation of a machine learning project for classifying chest X-ray images as Normal or COVID-19 positive. The project explores different CNN architectures and techniques to address class imbalance in medical imaging datasets.

## Project Overview

This project implements and compares two different approaches to COVID-19 detection from chest X-rays:

1. **Phase 1**: A baseline CNN implementation without optimization
2. **Phase 2**: An optimized approach using transfer learning, class balancing, and advanced training techniques

The goal is to identify the fastest, most cost-effective, and accurate method for clinical use.

## Dataset

The dataset consists of chest X-ray images divided into two classes:
- Normal (healthy patients)
- COVID-19 positive patients

The dataset has a significant class imbalance:
- Approximately 14% Normal X-rays
- Approximately 86% COVID-19 X-rays

The images are stored in the `data/raw/` directory with the following structure:
```
data/
├── raw/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── test/
│   └── train_labels.csv
```

## Environment Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
```
tensorflow>=2.4.0
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
pillow>=8.0.0
```

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure

```
covid_xray_model/
├── data/                      # Dataset directory
│   └── raw/                   # Raw X-ray images
├── scripts/                   # Python scripts
│   ├── restructure_train.py   # Script to restructure training data
│   ├── make_labels.py         # Script to generate labels
│   ├── run_experiments.py     # Original experiment script
│   ├── improved_experiments.py # Improved experiment script
│   └── fast_experiments.py    # Optimized fast experiment script
├── notebooks/                 # Jupyter notebooks (if any)
├── outputs/                   # Output directory for models and results
├── fast_training_results.png  # Training curves visualization
└── README.md                  # This file
```

## How to Run

### Phase 1: Baseline Implementation
The baseline implementation can be run using:

```bash
python scripts/run_experiments.py
```

This script tests different batch sizes and epoch values while tracking training/validation loss, accuracy, and execution time.

### Phase 2: Optimized Implementation
The optimized implementation can be run using:

```bash
python scripts/fast_experiments.py
```

This script implements:
- Transfer learning with MobileNetV2
- Class balancing techniques
- Optimal threshold selection
- Early stopping and learning rate scheduling

## Results

### Phase 1
- Achieved approximately 75.52% accuracy
- Training time: 9.83 minutes
- No clear relationship between batch size and accuracy
- Training loss increased with batch size
- Validation loss decreased with batch size

### Phase 2
- Achieved 89% overall accuracy
- Balanced performance across classes:
  - NORMAL: 98% recall, 83% precision, 90% F1-score
  - COVID: 80% recall, 98% precision, 88% F1-score
- Training time: 124.1 seconds (2.07 minutes)
- Confusion matrix:
  ```
  [[96  2]
   [20 81]]
  ```
- Successfully addressed class imbalance issues

## Key Findings

1. **Class Imbalance**: Addressing class imbalance is crucial for medical image classification. Without proper handling, models default to predicting the majority class.

2. **Transfer Learning**: Fine-tuning pre-trained models is effective for medical imaging, even when the source domain differs from the target domain.

3. **Error Patterns**: The model rarely misclassifies Normal X-rays as COVID (2 false positives) but more frequently misclassifies COVID X-rays as Normal (20 false negatives).

4. **Computational Efficiency**: The optimized approach achieved better accuracy in less time through architectural choices, image size reduction, and training optimizations.

## Contributors

- Alifyah: Phase 2 implementation (optimized approach)
- Kevin: Phase 1 implementation (baseline approach)
- Borna: Introduction and Literature Review

## Citation

If you use this code or findings in your research, please cite:

```
@misc{covid_xray_classification,
  author = {Alifyah and Kevin and Borna},
  title = {COVID-19 X-Ray Classification},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/alifyah-m/COSC4368_Project}}
}