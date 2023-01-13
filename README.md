## The Potential of Convolutional Neural Networks for Identifying Neural States based on Electrophysiological Signals: Code and Figures

This repository accompanies the manuscript 

> *The potential of convolutional neural networks for identifying neural states based on electrophysiological signals: experiments on synthetic and real patient data*


## Usage:

### 1. Installing packages & configuring environment:

The python (>=3.10) dependencies are listed in `requirements_gpu.yml` (a CUDA-capable system is required). We recommend installing them through anaconda or mamba:

```
conda env create -n <<environment_name>> --file requirements_gpu.yml 

conda activate <<environment_name>>
```

### 2. Downloading patient data:

Patient data can be downloaded at https://data.mrc.ox.ac.uk/lfp-et-dbs (DOI: 10.5287/bodleian:ZVNyvrw7R, creating a free account is required). This data should be placed into the `patient_data` folder.


### 3. Running code:

The repository is separated into a `nbs` folder and a `code` folder:

- The `nbs` folder includes notebooks with examples and the code used to generate all figures, which can be found here.
- The `code` folder implements synthetic data generation as well as model implementation and training. It includes two main files (`main_patients.py` and `main_synthetic.py`), which train models for patient data and synthetic data tasks respectively. Running these scripts takes some time (a total of over two weeks on a machine with two NVIDIA RTX-3090 GPUs). The scripts take in command-line arguments:

   Example: ```python main_patients.py --n_jobs 4 --accelerator gpu```

   To see all options: ```python main_synthetic.py --help```

   The script outputs are csv files containing performance metrics.

---

## License:
[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]


[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

&copy; The University of Oxford 2019 - 2023