import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
# from deel.lip.layers import (
#     SpectralDense,
#     SpectralConv2D,
#     ScaledL2NormPooling2D,
#     FrobeniusDense,
# )
# from deel.lip.model import Sequential
# from deel.lip.activations import GroupSort
# from deel.lip.losses import MulticlassHKR, MulticlassKR, HKR, HingeMargin
from keras.layers import Input, Flatten, Dense
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from deel import torchlip
import numpy as np
import keras.ops as K
import matplotlib.pyplot as plt
import torch
import pickle
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

from abstract_interpretation_tools import *
import sys
sys.path.append('/home/aws_install/robustess_project/lip_notebooks/')
from data_processing import load_data, select_data_for_radius_evaluation


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading model")
    pytorch_model = load_MNIST().to(device)

    print("loading dataset")
    # Define the directory and file paths
    output_dir = "./../benchmark_dataset_MNIST"
    images_path = os.path.join(output_dir, "images.pkl")
    targets_path = os.path.join(output_dir, "targets.pkl")

    # --- Load the Tensors ---
    print(f"Loading data from {output_dir}...")

    # Load the images tensor
    with open(images_path, 'rb') as f:
        images = pickle.load(f)

    # Load the targets tensor
    with open(targets_path, 'rb') as f:
        labels = pickle.load(f)
    
    images = images.to(device)
    labels = labels.to(device)

    print(pytorch_model(images[:1]))
    idx = 100

    print(images[idx:idx+1].shape)
    print("launching AutoLirpa")
    bounded_model = BoundedModule(pytorch_model, torch.ones_like(images[100:101]).to(device))
    bounded_model.eval()

    eps = 0.1
    norm = 2
    ptb = PerturbationLpNorm(norm = norm, eps = eps)
    # Input tensor is wrapped in a BoundedTensor object.
    bounded_image = BoundedTensor(images[100:101], ptb).to(device)
    print('Bounding method: backward (CROWN, DeepPoly)')

    # 2. Get the model's prediction on the clean image
    clean_logits = pytorch_model(images)
    pred_labels = torch.argmax(clean_logits, dim=1)

    # The same logic applies to batches
    
    true_label = labels[idx]
    num_classes = clean_logits.shape[1]

    # 3. Construct the specification matrix C for the logit differences
    # We want to compute the bounds for f_{true_label} - f_j for all j != true_label.

    # Create a list to hold the rows of the C matrix
    c_rows = []
    for j in range(num_classes):
        if j == true_label:
            continue  # Skip the f_true - f_true case
        
        # Create a row vector for f_true - f_j
        row = torch.zeros(1, num_classes)
        row[0, true_label] = 1.0
        row[0, j] = -1.0
        c_rows.append(row)

    # Stack the rows to form the final C matrix
    # The shape will be (num_classes - 1, num_classes)
    c_matrix = torch.cat(c_rows, dim=0)

    # Make sure the C matrix is on the correct device
    c_matrix = c_matrix.to(images.device)

    # 4. Compute the bounds using the C matrix
    # The output of compute_bounds will be the bounds on C * L, which are our logit differences.
    logit_diff_lb, logit_diff_ub = bounded_model.compute_bounds(
        x=(bounded_image,), 
        C=c_matrix.unsqueeze(0),
        method='CROWN'
    )


    print(logit_diff_lb, logit_diff_ub)

    # 5. Interpret the results
    # logit_diff_lb is a tensor of shape (batch_size, num_classes - 1)
    # It contains the lower bounds for [f_true - f_0, f_true - f_1, ..., f_true - f_{k-1}] (excluding f_true - f_true)

    print(f"Lower bounds on logit differences (f_{true_label} - f_j):")
    print(logit_diff_lb)

    # To certify robustness, we check if all these lower bounds are positive
    is_robust = torch.all(logit_diff_lb > 0)

    if is_robust:
        print(f"\nImage {idx} is CERTIFIED ROBUST for epsilon = {eps}")
        print(f"The minimum logit difference is {torch.min(logit_diff_lb).item()}")
    else:
        print(f"\nImage {idx} is NOT certified robust for epsilon = {eps}")
