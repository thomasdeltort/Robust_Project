# import os
# os.environ["KERAS_BACKEND"] = "torch"
# import keras
# import keras.ops as K
# from keras.layers import Input, Flatten, Dense
# from keras.optimizers import Adam
# from keras.metrics import BinaryAccuracy

# # from keras.models import Sequential
# from deel.lip.model import Sequential

# from deel.lip.layers import (
#     SpectralDense,
#     SpectralConv2D,
#     ScaledL2NormPooling2D,
#     FrobeniusDense,
# )
# from deel.lip.activations import GroupSort, GroupSort2
# from deel.lip.losses import HKR, KR, HingeMargin, MulticlassHKR, MulticlassKR

# import numpy as np
# import decomon
# import pandas as pd

# from lipschitz_optimization_tools import get_local_maximum_multiclass, echantillonner_boule_l2_simple, create_difference_model

# import sys
# sys.path.append('..')

# from radius_evaluation_tools import single_compute_relaxation_radius_multiclass


# from data_processing import load_data, select_data_for_radius_evaluation
# from radius_evaluation_tools import compute_binary_certificate, starting_point_dichotomy

# x_train, x_test, y_train, y_test, y_test_ord = load_data("MNIST")

# vanilla_model = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo0_vanilla_MNIST_channelfirst_False_disj_Neurons.keras")
# vanilla_model.compile(
#         # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
#         # note also in the case of lipschitz networks, more robustness require more parameters.
#         loss=MulticlassHKR(alpha=50, min_margin=0.05),
#         optimizer=keras.optimizers.Adam(1e-3),
#         metrics=["accuracy", MulticlassKR()],)

# images, labels, idx_list = select_data_for_radius_evaluation(x_test, y_test_ord, vanilla_model)

# total_points = images.shape[0]

# # Initialize the CSV file with column headers
# input_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST.csv"
# input_pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST.pkl"
# output_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_Decomon_MNIST_single_output_Relaxation.csv"
# output_pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_Decomon_MNIST_single_output_Relaxation.pkl"

# nb_pts = 100

# list_eps = []
# for i in range(total_points):
#     print(i)
#     eps_working = single_compute_relaxation_radius_multiclass(i, images, labels, vanilla_model, nb_pts)
#     list_eps.append(eps_working)

# df = pd.read_csv(input_csv_path)

# df["Relaxation"] = list_eps
# df.to_pickle(output_pkl_path)
# df.to_csv(output_csv_path, mode='a', header=False, index=False)



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
import csv  # Added import
import time # Added import

from lipschitz_optimization_tools import get_local_maximum_multiclass, echantillonner_boule_l2_simple, create_difference_model

import sys
sys.path.append('..')

from radius_evaluation_tools import single_compute_relaxation_radius_multiclass


from data_processing import load_data, select_data_for_radius_evaluation
from radius_evaluation_tools import compute_certificate, starting_point_dichotomy

if __name__ == "__main__":
    # --- Data and Model Loading ---
    x_train, x_test, y_train, y_test, y_test_ord = load_data("MNIST")

    vanilla_model = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo0_vanilla_MNIST_channelfirst_False_disj_Neurons.keras")
    vanilla_model.compile(
            loss=MulticlassHKR(alpha=50, min_margin=0.05),
            optimizer=keras.optimizers.Adam(1e-3),
            metrics=["accuracy", MulticlassKR()],)

    images, labels, idx_list = select_data_for_radius_evaluation(x_test, y_test_ord, vanilla_model)


    # 1. Define paths and parameters
    input_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST.csv"
    output_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_Decomon_MNIST_single_output_Relaxation_test.csv"
    output_pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_Decomon_MNIST_single_output_Relaxation_test.pkl"

    total_points = images.shape[0]
    nb_pts = 100

    # Compute Lipschitz Pessimistic Certificates
    print("Generating Certificates :")
    lip_radius = compute_certificate(images, vanilla_model)
    print(lip_radius)

    new_column_header = "Relaxation" # The header for your new column

    # 2. Load the input data and get existing headers
    # This now correctly assumes your input CSV has headers
    df_input = pd.read_csv(input_csv_path)
    output_headers = df_input.columns.tolist() + [new_column_header]

    # 3. Initialize the output CSV by writing the full header row
    with open(output_csv_path, 'w', newline='') as f_output:
        writer = csv.writer(f_output)
        writer.writerow(output_headers)

    # This list will be used to create the final pickle file
    list_for_pickle = []
    total_execution_time = 0.0

    # 4. Loop, calculate, and append each result to the CSV
    print("Starting calculations... results will be saved to CSV incrementally.")
    # Use `total_points` or set a smaller number for testing
    for i in range(total_points):
        start_time = time.time()
        print(f"Processing point {i+1}/{total_points}...")

        # Your calculation function remains the same
        eps_working = single_compute_relaxation_radius_multiclass(i, images, labels, vanilla_model, nb_pts, bounds='constant', input_shape=(1,28,28))
        
        # Store result for the pickle file
        list_for_pickle.append(eps_working)

        # Get the original data for the current row from the DataFrame
        original_row = df_input.iloc[i].tolist()
        
        # Append the new result to the CSV file immediately
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