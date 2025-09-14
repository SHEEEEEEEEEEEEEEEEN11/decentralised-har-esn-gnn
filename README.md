# Decentralised HAR with ESN + GCN-PPO

This repository contains the code and experiments for my MSc thesis on Decentralised Resource Sharing for Human Activity Recognition (HAR) Using Reservoir Computing and Graph Neural Networks. 

The code is organised to reproduce  an offline classifier (per-device vs shared ESN+LR) and system-level evaluations including routing, costs and multi-objective optimisation.

## Dataset

This project uses the [UCI-HAR dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).

UCI_HAR_Dataset/
├── train/
│   ├── Inertial Signals/
│   │   ├── body_acc_x_train.txt
│   │   ├── body_acc_y_train.txt
│   │   ├── body_acc_z_train.txt
│   │   ├── body_gyro_x_train.txt
│   │   ├── body_gyro_y_train.txt
│   │   ├── body_gyro_z_train.txt
│   │   ├── total_acc_x_train.txt
│   │   ├── total_acc_y_train.txt
│   │   ├── total_acc_z_train.txt
│   ├── X_train.txt
│   ├── y_train.txt
│   ├── subject_train.txt
├── test/
│   ├── Inertial Signals/
│   │   ├── body_acc_x_test.txt
│   │   ├── body_acc_y_test.txt
│   │   ├── body_acc_z_test.txt
│   │   ├── body_gyro_x_test.txt
│   │   ├── body_gyro_y_test.txt
│   │   ├── body_gyro_z_test.txt
│   │   ├── total_acc_x_test.txt
│   │   ├── total_acc_y_test.txt
│   │   ├── total_acc_z_test.txt
│   ├── X_test.txt
│   ├── y_test.txt
│   ├── subject_test.txt
├── activity_labels.txt
├── features.txt
├── features_info.txt
├── README.txt

