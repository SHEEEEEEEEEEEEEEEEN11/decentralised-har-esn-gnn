# Decentralised HAR with ESN + GCN-PPO

This repository contains the code and experiments for my MSc thesis on Decentralised Resource Sharing for Human Activity Recognition (HAR) Using Reservoir Computing and Graph Neural Networks. 

The code is organised to reproduce  an offline classifier (per-device vs shared ESN+LR) and system-level evaluations including routing, costs and multi-objective optimisation.

## Dataset

This project uses the [UCI-HAR dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).


UCI_HAR_Dataset/
├── train/ (inertial signals and labels for training)
├── test/ (inertial signals and labels for testing)
├── activity_labels.txt
├── features.txt
├── features_info.txt
├── README.txt




## Offline Classification

The script `per_device_vs_shared.py` trains and evaluates ESN+LR classifiers in two configurations:
- **Per-device**: training ESNs on Dirichlet-partitioned splits of the dataset to simulate six heterogeneous devices.  
- **Shared baseline**: using a single pooled reservoir (R = 1200) that is sliced into feasible sizes {400, 600, 800, 1200} and shared across all devices. 

The results are logged to CSV and plotted to compare accuracy across the different reservoir sizes.
