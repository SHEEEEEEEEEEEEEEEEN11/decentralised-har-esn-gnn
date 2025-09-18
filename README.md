# Decentralised HAR with ESN + GCN-PPO

This repository contains the code and experiments for an MSc thesis on **Decentralised Resource Sharing for Human Activity Recognition (HAR)** using Reservoir Computing (Echo State Networks, ESNs) and Graph Neural Networks (GCNs) with PPO-based routing policies.

The code is organised to reproduce:
- **Offline classification** (shared vs per-device ESN + Logistic Regression).
- **System-level evaluations** including routing, costs, and multi-objective optimisation.


## Dataset

This project uses the [UCI-HAR dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).


```
UCI_HAR_Dataset/
├── train/ (inertial signals and labels for training)
├── test/ (inertial signals and labels for testing)
├── activity_labels.txt
├── features.txt
├── features_info.txt
├── README.txt
```
## Installation

Install dependencies with:

pip install -r requirements.txt


```
## Repository Structure

offline_classification/ # Standalone classification (no routing)
routing/
per-device-budget.py # Train = stochastic PPO, Eval = budget-aware heuristic
per-device-greedy.py # Greedy baseline (fixed best-R per device)
per-device-learned.py # Train = stochastic PPO, Eval = greedy masked argmax
shared-budget.py # Shared reservoir, budget-aware heuristic
shared-greedy.py # Shared reservoir, greedy baseline
shared-learned.py # Shared reservoir, stochastic PPO + greedy masked argmax
README.md # Project overview and instructions
requirements.txt # Python dependencies
```
Each script can be run independently. Outputs are saved into the configured `OUT_DIR` inside each script.

## Running Experiments
```
All experiments assume that the **UCI HAR Dataset** has been downloaded and placed under `DATA_DIR` (default is `/content/drive/MyDrive/UCI_HAR_Dataset`).  
Each script saves artefacts and logs automatically into the `artifacts/` or `thesis/system_` folders.  

 1. Offline Classification (baseline classifiers)

Train and evaluate the classification models only (no routing).  
These scripts save ESN → Logistic Regression artefacts for downstream routing.  

 Shared ESN classifier
python offline_classification/shared_classification.py --data_dir /path/to/UCI_HAR_Dataset --save_path artifacts/shared_esn.pkl

 Per-device ESN classifiers (Dirichlet α=5)
python offline_classification/per_device_classification.py --data_dir /path/to/UCI_HAR_Dataset --save_path artifacts/per_device_alpha5.pkl

After building the offline artefacts, run the routing policies.
Each script trains a router (stochastic PPO) and evaluates using either a learned or heuristic strategy.

 2. Routing Policies

Per-device policies
python routing/per-device-learned.py
python routing/per-device-budget.py
python routing/per-device-greedy.py

Shared-reservoir policies
python routing/shared-learned.py
python routing/shared-budget.py
python routing/shared-greedy.py
```

Each script produces:
train_metrics.csv → PPO training logs (if applicable)
window_log.csv → full per-window evaluation results
Headline metrics printed in the console
