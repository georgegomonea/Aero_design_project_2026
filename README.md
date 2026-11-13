# Aero Project Environment Setup

This project uses a Conda environment to manage dependencies such as NumPy, SciPy, Matplotlib, and FAST-OAD. The environment configuration is defined in the `environment.yml` file.

## Prerequisites

Ensure you have a Conda distribution installed on your system. Using [Miniforge](https://github.com/conda-forge/miniforge) (which defaults to `mamba` and `conda-forge`) is highly recommended for faster dependency solving.

## Setup Instructions

**1. Create the environment**
Navigate your terminal to the directory containing the `environment.yml` file and run the following command:

```bash
conda env create -f environment.yml