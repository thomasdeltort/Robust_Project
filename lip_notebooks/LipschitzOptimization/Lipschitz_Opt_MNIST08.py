import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import keras.ops as K
from keras.layers import Input, Flatten, Dense
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
import csv
import time

from lipschitz_optimization_tools import get_local_maximum, echantillonner_boule_l2_simple

import sys
sys.path.append('..')

from radius_evaluation_tools import single_compute_relaxation_radius


from data_processing import load_data, select_data_for_radius_evaluation_MNIST08
from radius_evaluation_tools import compute_binary_certificate, starting_point_dichotomy

if __name__ == "__main__":
    x_train, x_test, y_train, y_test, y_test_ord = load_data("MNIST08")

    model_path = "/home/aws_install/robustess_project/lip_models/demo3_FC_vanilla_MNIST08_channelfirst_False_disj_Neurons_single_output.keras"
    model = keras.models.load_model(model_path)
    model.compile(
    
        loss=HKR(
            alpha=10.0, min_margin=1.0
        ),  # HKR stands for the hinge regularized KR loss
        metrics=[
            # KR,  # shows the KR term of the loss
            HingeMargin(min_margin=1.0),  # shows the hinge term of the loss
        ],
        optimizer=Adam(learning_rate=0.001),)

    model_bis = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo3_FC_vanilla_MNIST08_channelfirst_False_disj_Neurons_single_output_converted_4logits.keras")
    model_bis.compile(
            # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
            # note also in the case of lipschitz networks, more robustness require more parameters.
            loss=MulticlassHKR(alpha=100, min_margin=0.25),
            optimizer=Adam(1e-4),
            metrics=["accuracy", MulticlassKR()],)

    images, labels, idx_list = select_data_for_radius_evaluation_MNIST08(x_test, y_test_ord, model_bis)

    total_points = images.shape[0]
    # # 1. Define the headers for your columns
    # input_headers = [
    #     "Index",
    #     "Label_GT",
    #     "Predicted_Label",
    #     "Lipschitz_Constant",
    #     "Robust_Epsilon",
    #     "Adv_Epsilon_AA",
    #     "Adv_Epsilon_PGD",
    #     # "Convex_Relaxation_Epsilon",
    # ]
    new_column_header = "Constant_Relaxation_Epsilon"
    # The full list of headers for the output file
    # output_headers = input_headers + [new_column_header]


    # Paths for input and output files
    input_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST08_single_output.csv"
    output_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST08_single_output_Relaxation_test.csv"
    output_pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST08_single_output_Relaxation_test.pkl"

    # 2. Load the original input data to read from
    df_input = pd.read_csv(input_csv_path)
    output_headers = df_input.columns.tolist() + [new_column_header]
    # Parameters for calculation
    total_points = images.shape[0]
    nb_pts = 100

    # Compute Lipschitz Pessimistic Certificates
    print("Generating Certificates :")
    lip_radius = compute_binary_certificate(images, model)

    # 3. Initialize the output CSV file by writing the header row
    # This opens the file in write mode 'w', creating it and writing only the header
    with open(output_csv_path, 'w', newline='') as f_output:
        writer = csv.writer(f_output)
        writer.writerow(output_headers)

    # This list will be used to create the final pickle file
    list_for_pickle = []
    total_execution_time = 0.0

    # 4. Loop, calculate, and APPEND each result to the CSV
    print("Starting calculations... results will be saved to CSV incrementally.")
    # Adjust the range for the number of points you want to process
    for i in range(total_points):
        start_time = time.time()
        print(f"Processing point {i+1}/{total_points}...")

        # Your calculation function remains the same
        eps_working = single_compute_relaxation_radius(i, images, labels, model, nb_pts, bounds="constant")
        
        # Store result for the pickle file later
        list_for_pickle.append(eps_working)

        # Get the original data for the current row
        original_row = df_input.iloc[i].tolist()
        
        # --- Append to CSV immediately ---
        # Open the output file in append mode 'a'
        with open(output_csv_path, 'a', newline='') as f_output:
            writer = csv.writer(f_output)
            # Create the full row and write it to the file
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

    print(f"\n✅ Processing complete. Incremental results saved to:\nCSV: {output_csv_path}")
    print(f"Final complete dataset saved to:\nPKL: {output_pkl_path}")

    if total_points > 0:
        average_time = total_execution_time / total_points
        print(f"\nAverage time per point: {average_time:.2f} seconds.")



