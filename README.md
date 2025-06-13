# DCGAN-Based Data Augmentation for Image Classification

## Overview

This repository provides an interactive Jupyter Notebook pipeline (`gans_for_data_augmentation.ipynb`) that walks through data preprocessing, DCGAN training, and synthetic image generation to augment your dataset and boost classifier performance.

Key features:

* End-to-end DCGAN workflow implemented in a single notebook
* Automated download and preprocessing of the Oxford Flowers 102 dataset
* Canonical Generator & Discriminator definition with weight initialization
* Training loop visualization with loss curves and sample grids
* Generation and packaging of synthetic images for downstream use

## Repository Structure

```
├── data/
│   ├── raw/                # Downloaded archives and extracted images
│   └── processed/          # Organized 64×64 RGB images by class
│
├── experiments/            # Saved model checkpoints and sample image grids
│
├── outputs/                # Generated synthetic images and ZIP archive
│
├── gans_for_data_augmentation.ipynb  # Jupyter Notebook with full DCGAN pipeline
│
├── requirements.txt        # Python dependencies
│
└── README.md               # This overview and usage instructions
```

## Prerequisites

* Python 3.8+
* JupyterLab or Jupyter Notebook
* PyTorch 1.10+ (with CUDA support for GPU training)
* torchvision, numpy, scipy, matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. **Launch the Notebook**

   ```bash
   jupyter lab GANs_for_Data_Augmentation.ipynb
   ```
2. **Run All Cells**
   The notebook is organized into sections:

   * **Setup & Imports**: define dependencies and utility functions
   * **Data Download & Preprocessing**: fetch Oxford Flowers 102, resize, normalize, and organize
   * **Model Definitions**: implement DCGAN Generator and Discriminator architectures
   * **Training Loop**: train GAN, log losses, and save sample image grids
   * **Image Generation**: sample latent vectors, generate synthetic images, and archive results
3. **Review Outputs**

   * Check `experiments/` for model checkpoints and example grids.
   * Find generated images in `outputs/augmented_images.zip` for integration with your classifier.

## References

1. **Goodfellow et al.** Generative Adversarial Nets (2014): [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)
2. **Radford et al.** Deep Convolutional GANs (2015): [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)
3. **Arjovsky et al.** Wasserstein GAN (2017): [https://arxiv.org/abs/1701.07875](https://arxiv.org/abs/1701.07875)
4. **Gulrajani et al.** Improved WGAN (2017): [https://arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)
5. **eriklindernoren/PyTorch-GAN**: [https://github.com/eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
6. **hindupuravinash/the-gan-zoo**: [https://github.com/hindupuravinash/the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo)

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
