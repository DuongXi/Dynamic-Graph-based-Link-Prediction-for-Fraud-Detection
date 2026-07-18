# Dynamic Graph-based Link Prediction for Fraud Detection Data

This repository contains a temporal graph learning workflow built around neural process models for dynamic link prediction. The main pipeline includes:

- preprocessing raw edge data into a format suitable for training,
- building graph index files and temporal split labels,
- configuring model variants through YAML files,
- training and evaluating the model from the command line.

## Project structure

- data_process__pt.py: converts raw edge CSV files into dataframe and feature files.
- gen_graph_custom.py: builds adjacency index files and time-based train/validation/test splits.
- train_np.py: training and evaluation entry point for the model.
- config/: contains YAML configuration files for different model variants (APAN, DySAT, JODIE, TGAT, TGN).
- sampler_core.cpp and setup.py: C++ sampler extension used during training.

## Typical workflow

1. Prepare the dataset in the expected directory layout under DATA/.
2. Run preprocessing:
   ```bash
   python data_process__pt.py --data <dataset_name> --bipartite
   ```
3. Build graph index files:
   ```bash
   python gen_graph_custom.py --data <dataset_name> --raw_data <raw_file_name>
   ```
4. Train the model:
   ```bash
   python train_np.py --path <data_path> --data <dataset_name> --config config/TGN.yml
   ```

## Configuration notes

The YAML files in the config folder define sampling behavior, memory modules, graph neural network options, and training hyperparameters. The files are a good starting point for tuning a model without modifying the training code directly.

## Requirements

Install the Python dependencies required by the project before running experiments. A typical environment includes:

- Python 3.8+
- PyTorch
- DGL
- NumPy
- pandas
- scikit-learn
- tqdm
- pybind11

## Notes

The codebase originally referenced a Google Drive-based data path in the preprocessing script. If you are running locally, update the data path in the script to match your own dataset location.

## Acknowlement
This repo is mainly based on [amazon-science/tgl](https://github.com/amazon-science/tgl) and [RManLuo/GSNOP](https://github.com/RManLuo/GSNOP). We thank the authors for their great works.