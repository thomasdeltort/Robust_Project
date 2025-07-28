import torch
import pandas as pd
import numpy as np
# import pdb
import yaml
import os
import pickle



from data_processing_torch import *
from radius_evaluation_tools_torch import *

from notebooks_creation_models.VGG_Arthur import *

import sys
sys.path.append("..")
import liresnet.models as models

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading data :")
    # Load configuration file
    with open('./notebooks_creation_models/config.yaml', 'r') as f:
        # maybe dupplicate config
        cfg = yaml.safe_load(f)
    _, test_loader = load_cifar10(cfg)

    print("loading model :")
    weights = torch.load('/home/aws_install/robustess_project/lip_models/cifar10-12x512_799.pth').get('backbone')
    with open('/home/aws_install/robustess_project/liresnet/configs/cifar10.yaml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    model_cfg = cfg['model']
    dataset_cfg = cfg['dataset']
    gloro_cfg = cfg['gloro']
    model = models.GloroNet(**model_cfg, **dataset_cfg).to(device)
    model.load_state_dict(weights)
    model.eval()

   
    output_dir = "./benchmark_dataset"
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
    lip_radius = compute_certificate_LiResNet(images, model).to(device)
    # pdb.set_trace()
    # Initialize the CSV file with column headers
    columns = ["Index", "Label_GT", "Predicted_Label", "Lipschitz_Constant", "Robust_Epsilon", "Adv_Epsilon_AA", "Adv_Epsilon_PGD"]
    csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_CIFAR10_liresnet.csv"
    pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_CIFAR10_liresnet.pkl"

    # Create an empty file with headers
    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
    
    df_list = []

    for i in range(total_points):
        eps_pgd = single_compute_optimistic_radius_PGD(i, images, labels, lip_radius, model, n_iter=10)
        eps_aa = single_compute_optimistic_radius_AA(i,images, labels, lip_radius, model, n_iter=10)
        print("Point ", i, "attaques trouvées :", eps_pgd, eps_aa)
        
        # Create a row
        row = {
            "Index": i,
            "Label_GT": labels[i].detach().cpu().numpy(),
            "Predicted_Label": np.argmax(model(images[i:i+1]).detach().cpu().numpy(), axis=1)[0],
            "Lipschitz_Constant": 1.0,
            "Robust_Epsilon": lip_radius[i].detach().cpu().numpy(),
            "Adv_Epsilon_AA": eps_aa.item(),
            "Adv_Epsilon_PGD": eps_pgd.item()}
        
        # Append to CSV file without rewriting the header
        pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)
        
        # Append to the list for Pickle
        df_list.append(row)
        
        # Save to Pickle at each iteration
        pd.DataFrame(df_list).to_pickle(pkl_path)