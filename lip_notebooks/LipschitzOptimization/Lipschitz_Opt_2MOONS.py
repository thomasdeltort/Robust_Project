import os
# os.environ["KERAS_BACKEND"] = "torch"
# import keras
# import keras.ops as K
# from keras.layers import Input, Flatten, Dense, TorchModuleWrapper
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

import numpy as np
# import decomon
import pandas as pd
import csv  # Added import
import time # Added import
import yaml
import torch
import pickle
from lipschitz_optimization_tools_torch import get_local_maximum_multiclass, echantillonner_boule_l2_simple

import sys
sys.path.append('..')

from radius_evaluation_tools_torch import single_compute_relaxation_radius_accuracy
from data_processing import load_data, select_data_for_radius_evaluation
from radius_evaluation_tools_torch import compute_binary_certificate, starting_point_dichotomy
from notebooks_creation_models.VGG_Arthur import *
from radius_evaluation_tools_torch import compute_binary_certificate

class ModelWrapper(torch.nn.Module):
    """
    Un wrapper qui prend un input de taille (batch, 2, 1, 1),
    l'aplatit en (batch, 2), et l'applique au modèle original.
    """
    def __init__(self, model_to_wrap):
        super().__init__()
        # On stocke le modèle original
        self.model = model_to_wrap

    def forward(self, x):
        # Shape de x en entrée : (batch, 2, 1, 1)
        
        # On aplatit le tenseur à partir de la deuxième dimension
        # Le premier argument '1' de flatten indique de ne pas toucher à la dimension du batch
        x_flattened = torch.flatten(x, 1)
        
        # Shape de x_flattened : (batch, 2)
        
        # On applique le modèle original sur le tenseur aplati
        return self.model(x_flattened)
    
def load_models(device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Other Lipschitz activations are ReLU, MaxMin, GroupSort2, GroupSort.
    wass = torchlip.Sequential(
        torchlip.SpectralLinear(2, 256),
        torchlip.GroupSort2(),
        torchlip.SpectralLinear(256, 128),
        torchlip.GroupSort2(),
        torchlip.SpectralLinear(128, 64),
        torchlip.GroupSort2(),
        torchlip.SpectralLinear(64, 1, bias=False),
    ).to(device)

    wass = wass.vanilla_export()
    # Récupérer la dernière couche de l'ancien modèle
    original_last_layer = wass[-1]
    # Récupérer toutes les couches sauf la dernière
    all_but_last_layers = list(wass.children())[:-1]

    # Créer la nouvelle couche Dense (Linear en PyTorch)
    # L'entrée (in_features) doit correspondre à la sortie de la couche précédente
    new_out_features = 4
    new_linear_layer = nn.Linear(
        in_features=original_last_layer.in_features, # Sera 16
        out_features=new_out_features,               # Sera 4
        bias=False
    )

    # Créer le nouveau modèle en ajoutant la nouvelle couche à la fin
    new_model = nn.Sequential(*all_but_last_layers, new_linear_layer, nn.Flatten())

    print("Architecture du nouveau modèle:")
    print(new_model)
    print("-" * 30)


    # --------------------------------------------------------------------------
    # 3. Copier et modifier les poids
    # --------------------------------------------------------------------------

    # Utiliser torch.no_grad() est essentiel pour manipuler les poids
    # manuellement sans que l'autograd ne suive ces opérations.
    with torch.no_grad():
        # Récupérer les poids et biais de la couche d'origine
        original_weights = original_last_layer.weight.data # Shape: (2, 16)
        # original_bias = original_last_layer.bias.data     # Shape: (2,)

        # Créer les nouveaux tenseurs de poids et de biais, initialisés à zéro
        # Notez la forme (out, in) pour les poids en PyTorch !
        w_temp = torch.zeros_like(new_linear_layer.weight.data) # Shape: (4, 16)
        # b_temp = torch.zeros_like(new_linear_layer.bias.data)   # Shape: (4,)

        # Copier les poids d'origine dans les premières lignes du nouveau tenseur de poids
        # On copie les 2 lignes du tenseur (2, 16) dans les 2 premières lignes de (4, 16)
        w_temp[0:1, :] = -original_weights
        w_temp[1:2, :] = original_weights

        # # Copier les biais d'origine
        # b_temp[:original_bias.shape[0]] = original_bias
        
        # # Appliquer la modification spécifique du biais
        # b_temp[2:] = -10000

        # Assigner les nouveaux poids et biais à la nouvelle couche
        new_linear_layer.weight.data = w_temp
        # new_linear_layer.bias.data = b_temp

    new_model.to(device)
    new_model.eval()
    wrapped_model = ModelWrapper(new_model).to(device)
    wrapped_model.eval()
    return wass, wrapped_model

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading model :")
    model, wrapped_model = load_models(device)
    model.load_state_dict(torch.load("/home/aws_install/robustess_project/lip_models/FC_2MOONS_Lip.pt", weights_only=True))
    model.eval()
    wrapped_model.load_state_dict(torch.load("/home/aws_install/robustess_project/lip_models/FC_2MOONS_WrappedModel_Lip.pt", weights_only=True))
    wrapped_model.eval()

    # layer_torch = TorchModuleWrapper(model)
    # k_model = keras.models.Sequential([Input((2,)), layer_torch])

    print("Loading Sample :")
   
    # Define the directory and file paths
    output_dir = "./../benchmark_dataset_2MOONS"
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
    lip_radius = compute_binary_certificate(images, model)

    # 1. Define paths and parameters
    input_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_2MOONS.csv"
    output_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_2MOONS_Relaxation_complete.csv"
    output_pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_2MOONS_Relaxation_complete.pkl"

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
        eps_working = single_compute_relaxation_radius_accuracy(i, images, labels, model, nb_pts, device=device, input_shape=(2,), lip_certificate=lip_radius)
        
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

