# Deep Self-Learning From Noisy Labels

This repository implements the Deep Self-Learning (DSL) framework for robust image classification in the presence of noisy labels, as proposed by Han et al. in their ICCV 2019 paper: [Deep Self-Learning From Noisy Labels](https://openaccess.thecvf.com/content_ICCV_2019/papers/Han_Deep_Self-Learning_From_Noisy_Labels_ICCV_2019_paper.pdf).

## Overview

Training deep neural networks with noisy labels often leads to performance degradation. The DSL framework addresses this challenge by introducing an iterative self-learning approach that:

- **Prototype Selection**: Identifies high-confidence samples (prototypes) based on data density and similarity metrics.
- **Label Correction**: Utilizes these prototypes to correct noisy labels in the dataset.
- **Iterative Training**: Refines the model through successive training iterations using the corrected labels.

This method does not rely on assumptions about noise distribution or require additional clean supervision, making it effective for real-world noisy datasets like Clothing1M and Food101-N.

## Repository Contents

- `Clothing1M.ipynb`: Implementation of the DSL framework on the Clothing1M dataset.
- `Food101-RP-Approach.ipynb`: Application of DSL to the Food101-N dataset using a representative prototype approach.
- `food-image-classification.ipynb`: General image classification pipeline incorporating DSL techniques.

## Datasets

- **Clothing1M**: Contains 1 million clothing images with noisy labels across 14 categories.
- **Food101-N**: Comprises 310,009 food images with noisy labels spanning 101 categories.

Please ensure you have access to these datasets and adjust the data paths in the notebooks accordingly.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Jupyter Notebook
- PyTorch
- NumPy
- Pandas
- Matplotlib

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/DeRohan/Deep-Self-Learning-From-Noisy-Labels.git
   cd Deep-Self-Learning-From-Noisy-Labels
