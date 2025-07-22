import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from deel import torchlip
# import pdb
import yaml
import os 
import pickle


from data_processing_torch import *
from radius_evaluation_tools_torch import *

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

    print("loading data :")
    # # Load configuration file
    # with open('./notebooks_creation_models/config.yaml', 'r') as f:
    #     cfg = yaml.safe_load(f)
    # _, test_loader = load_cifar10(cfg)

    print("loading model :")
    model, wrapped_model = load_models(device)
    model.load_state_dict(torch.load("./../lip_models/FC_2MOONS_Lip.pt", weights_only=False))
    model.eval()
    wrapped_model.load_state_dict(torch.load("./../lip_models/FC_2MOONS_WrappedModel_Lip.pt", weights_only=False))
    wrapped_model.eval()

    print("Loading Sample :")
    # images, labels = select_data_for_radius_evaluation(test_loader, test_loader.dataset, model, schedulefree=True)
    # images = images.to(device)
    # labels = labels.to(device)
    # Define the directory and file paths
    output_dir = "./benchmark_dataset_2MOONS"
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

    total_points = images.shape[0]


    print("Generating Certificates :")
    lip_radius = compute_binary_certificate(images, model)

    # Initialize the CSV file with column headers
    columns = ["Index", "Label_GT", "Predicted_Label", "Lipschitz_Constant", "Robust_Epsilon", "Adv_Epsilon_AA", "Adv_Epsilon_PGD"]
    csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_2MOONS.csv"
    pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_2MOONS.pkl"

    # Create an empty file with headers
    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
    
    df_list = []
    adv_images = []
    list_eps_pgd = []

    for i in range(total_points):
        # we have to send labels from {-1,1} to {0,1}
        labels_01 = (labels/2 + 0.5).long()

        eps_pgd, adv_image = single_compute_optimistic_radius_PGD(i, images.unsqueeze(-1).unsqueeze(-1), labels_01, lip_radius, wrapped_model, n_iter=10)
        # eps_aa = single_compute_optimistic_radius_AA_binary(i,images, labels, lip_radius, model, n_iter=10)
        print("Point ", i, "attaques trouvées :", eps_pgd)
        # Create a row
        # row = {
        #     "Index": i,
        #     "Label_GT": (labels[i]/2 + 0.5).long().item(),
        #     "Predicted_Label": np.argmax(model(images[i:i+1]).detach().cpu().numpy(), axis=1)[0],
        #     "Lipschitz_Constant": 1.0,
        #     "Robust_Epsilon": lip_radius[i].detach().cpu().numpy(),
        #     "Adv_Epsilon_AA": 0.0,
        #     "Adv_Epsilon_PGD": eps_pgd.item()}

        adv_images.append(adv_image)
        list_eps_pgd.append(eps_pgd)

        # # Append to CSV file without rewriting the header
        # pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)
        
        # # Append to the list for Pickle
        # df_list.append(row)
        
        # # Save to Pickle at each iteration
        # pd.DataFrame(df_list).to_pickle(pkl_path)
    adv_images = np.array(adv_images)
    list_eps_pgd = np.array(list_eps_pgd)

    tuple_data = (lip_radius, list_eps_pgd, adv_images, images)
    print(tuple_data)