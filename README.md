# InfoQGAN

InfoQGAN is the quantum version of InfoGAN. This repository contains code for training and testing the InfoQGAN model compare to the QGAN, InfoGAN, GAN.

## Getting Started

You will need to install the appropriate packages via requirements.txt.

Before getting started, create a `2d_runs` and `iris_runs` folder in the root directory. This folder is used to store TensorBoard data and training logs, and it is ignored by git.

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

- Python 3.10 or higher (**This code is written based on Python 3.12**)
- The required packages are defined in the `requirements.txt` file.


To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```

### How to Run
If you want to train for 2D dataset (please make `2d_runs` folder before you run):
```bash
python 2d_train.py --model_type InfoQGAN
```

or

```bash
python 2d_train_classical.py --model_type InfoGAN
```



If you want to train with IRIS (please make `iris_runs` folder before you run):
```bash
python iris_train.py --model_type InfoQGAN
```

or

```bash
python iris_train_classical.py --model_type InfoGAN
```
