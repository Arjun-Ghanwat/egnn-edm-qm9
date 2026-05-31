# egnn-edm-qm9
Replication and implementation of E(n)-Equivariant Graph Neural Networks (EGNN) and Equivariant Diffusion Models (EDM) for 3D molecular generation on the QM9 dataset. This repository reproduces core architectures, training procedures, and molecular generation experiments from the original papers.
# EGNN & EDM on QM9

## Overview

This repository contains a replication of:

- E(n)-Equivariant Graph Neural Networks (EGNN)
- Equivariant Diffusion Models (EDM)

for 3D molecular generation on the QM9 dataset.

The implementation follows the original papers and aims to reproduce the core architecture, training pipeline, and molecular generation process.

## Papers

1. EGNN (Satorras et al., 2021)
2. EDM (Hoogeboom et al., 2022)

## Dataset

QM9 Dataset:
https://www.kaggle.com/datasets/zaharch/quantum-machine-9-aka-qm9

The dataset contains approximately 134,000 small organic molecules with atomic coordinates and molecular properties.

## Contents

- Data preprocessing
- EGNN architecture
- Equivariant diffusion model
- Training and validation
- Molecular generation
- Result visualization

## Usage

Open and run:

egnn-and-edm.ipynb

## Results

The notebook doesnt have any result due to lack of computational power , but if u run it u will get percentage of test data it got correct . it also give you Validation and Training error after each epoch . You can change epochs and batch size as per GPU Size . 

## References

Satorras et al., 2021. E(n)-Equivariant Graph Neural Networks.

Hoogeboom et al., 2022. Equivariant Diffusion for Molecule Generation in 3D.
