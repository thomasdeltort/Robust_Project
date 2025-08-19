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

    print("launching AutoLirpa")
    bounded_model = BoundedModule(pytorch_model, torch.ones_like(images[100:101]).to(device))
    bounded_model.eval()

    eps = 0.01
    norm = 2
    ptb = PerturbationLpNorm(norm = norm, eps = eps)
    # Input tensor is wrapped in a BoundedTensor object.
    bounded_image = BoundedTensor(images[100:101], ptb).to(device)
    print('Bounding method: backward (CROWN, DeepPoly)')
    with torch.no_grad():  # If gradients of the bounds are not needed, we can use no_grad to save memory.
        lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method='CROWN-IBP')
    print(lb, ub)

    # print(torch.argmax(pytorch_model(images)), labels)