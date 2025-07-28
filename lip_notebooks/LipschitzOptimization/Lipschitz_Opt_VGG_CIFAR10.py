import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import keras.ops as K
from keras.layers import Input, Flatten, Dense, TorchModuleWrapper
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy

# from keras.models import Sequential
from deel.lip.model import Sequential

from deel.lip.layers import (
    SpectralDense,
    SpectralConv2D,
    ScaledL2NormPooling2D,
    FrobeniusDense,
)
from deel.lip.activations import GroupSort, GroupSort2
from deel.lip.losses import HKR, KR, HingeMargin, MulticlassHKR, MulticlassKR

import numpy as np
import decomon
import pandas as pd
import csv  # Added import
import time # Added import
import yaml
import torch
import pickle
from lipschitz_optimization_tools import get_local_maximum_multiclass, echantillonner_boule_l2_simple, create_difference_model

import sys
sys.path.append('..')

from radius_evaluation_tools import single_compute_relaxation_radius_multiclass_accuracy
from data_processing import load_data, select_data_for_radius_evaluation
from radius_evaluation_tools import compute_binary_certificate, starting_point_dichotomy
from notebooks_creation_models.VGG_Arthur import *
from radius_evaluation_tools_torch import compute_certificate

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading model :")
    model = load_model().vanilla_export().to(device)
    model.load_state_dict(torch.load('/home/aws_install/robustess_project/lip_notebooks/notebooks_creation_models/Vgg_lip_multisteplr_van.pt', weights_only=True))
    model.eval()

    layer_torch = TorchModuleWrapper(model)
    k_model = keras.models.Sequential([Input((3,32,32)), layer_torch])

    print("loading data :")
    output_dir = "./../benchmark_dataset"
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

    print("Generating Certificates :")
    lip_radius = compute_certificate(images, model)

    # 1. Define paths and parameters
    input_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_CIFAR10_VGG.csv"
    output_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_CIFAR10_VGG_Relaxation.csv"
    output_pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_CIFAR10_VGG_Relaxation.pkl"

    total_points = images.shape[0]
    nb_pts = 100
    new_column_header = "Relaxation"

    # 2. Load the input data and get existing headers
    df_input = pd.read_csv(input_csv_path)
    output_headers = df_input.columns.tolist() + [new_column_header]

    # 3. Initialize the output CSV by writing the header row
    with open(output_csv_path, 'w', newline='') as f_output:
        writer = csv.writer(f_output)
        writer.writerow(output_headers)

    # This list will be used to create the final pickle file
    list_for_pickle = []
    total_execution_time = 0.0

    # 4. Loop, calculate, and append each result to the CSV
    print("🚀 Starting calculations... results will be saved to CSV incrementally.")
    for i in range(total_points):
        start_time = time.time()
        print(f"Processing point {i+1}/{total_points}...")

        # Your calculation function
        eps_working = single_compute_relaxation_radius_multiclass_accuracy(i, images, labels, k_model, nb_pts, input_shape=(3,32,32), lip_certificate=lip_radius)
        
        # Store result for the pickle file
        list_for_pickle.append(eps_working)

        # Get the original data for the current row
        original_row = df_input.iloc[i].tolist()
        
        # Append the new result to the CSV file
        with open(output_csv_path, 'a', newline='') as f_output:
            writer = csv.writer(f_output)
            row_to_write = original_row + [eps_working]
            writer.writerow(row_to_write)

        end_time = time.time()
        duration = end_time - start_time
        total_execution_time += duration
        print(f"-> Point {i+1} saved. Time for this point: {duration:.2f} seconds.")

    # 5. After the loop, create and save the final pickle file
    print("\nAll points processed. Saving final pickle file...")
    df_input[new_column_header] = pd.Series(list_for_pickle)
    df_input.to_pickle(output_pkl_path)

    print(f"\n✅ Processing complete. Incremental results saved to:\n   CSV: {output_csv_path}")
    print(f"Final complete dataset saved to:\n   PKL: {output_pkl_path}")

    if total_points > 0:
        average_time = total_execution_time / total_points
        print(f"\nAverage time per point: {average_time:.2f} seconds.")

