# InfoQGAN

InfoQGAN is the quantum version of InfoGAN. This repository contains code for training and testing the InfoQGAN model compare to the QGAN, InfoGAN, GAN.

## Getting Started

You will need to install the appropriate packages via requirements.txt.

Before getting started, create a `runs` folder in the root directory. This folder is used to store TensorBoard data and training logs, and it is ignored by git.

The model is small enough to run without a GPU.

## Directory Structure

```bash
INFOQGAN/
├── data/                           # Data files
│   ├── 2D/                         # 2D training data
│   ├── IRIS/                       # IRIS augmentation data
├── modules/                        # Models & utilities
│   ├── Discriminator.py            # Discriminator model
│   ├── MINE.py                     # Mutual info estimator (MINE)
│   ├── Generator.py                # Classical generator
│   ├── QGenerator.py               # Quantum generator
│   └── utils.py                    # Utility functions
├── .gitignore                      
├── 2D_train.py                     # Train QGAN/InfoQGAN on 2D data
├── 2D_train_classical.py           # Train GAN/InfoGAN on 2D data
├── 2d_custom_shape.ipynb           # Custom 2D shapes notebook
├── iris_train.py                   # Train QGAN/InfoQGAN on IRIS data
├── iris_train_classical.py         # Train GAN/InfoGAN on IRIS data
├── iris_augment_evaluation.ipynb   # Evaluate IRIS augmentation performance
├── README.md                       # Project overview
├── requirements.txt                # Required packages (Python 3.10)


```

### Requirements

- Python 3.10 or higher (**This code is written based on Python 3.10**)
- The required packages are defined in the `requirements.txt` file.

To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```

### How to Run
If you want to train with InfoQGAN:
```bash
python mnist_train.py --model_type InfoQGAN --DIGITS_STR 0123456789 --DIGIT 1 --G_lr 0.01 --M_lr 0.0001 --D_lr 0.001 --coeff 0.05 --epochs 300 --latent_dim 16 --num_images_per_class 2000
```

If you want to train with QGAN:
```bash
python mnist_train.py --model_type QGAN --DIGITS_STR 0123456789 --DIGIT 1 --G_lr 0.01 --M_lr 0.0001 --D_lr 0.001 --coeff 0.05 --epochs 300 --latent_dim 16 --num_images_per_class 2000
```