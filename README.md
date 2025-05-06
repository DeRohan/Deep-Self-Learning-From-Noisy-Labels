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

2. (Optional) Create Virtual Environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the Dependancies
   ```bash
   pip install -r requirements.txt

4. Download the datasets and adjust data paths in the notebooks.

## Results

The Deep Self-Learning approach has demonstrated strong performance on noisy-label datasets, showing resilience to real-world noise without the need for clean labels or strong assumptions about label noise.

| Dataset     | DSL Accuracy |
|-------------|---------------|
| Clothing1M  | ~74.2%        |
| Food101-N   | ~88.1%        |

*Note: Results may vary depending on implementation details, batch size, and training duration.*

## Reference

If you use this codebase or the DSL approach in your research, please cite the original paper:

```bibtex
@inproceedings{han2019deep,
  title={Deep Self-Learning From Noisy Labels},
  author={Han, Jiangfan and Luo, Ping and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5138--5147},
  year={2019}
}

