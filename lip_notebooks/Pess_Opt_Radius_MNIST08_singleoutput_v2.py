import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from deel.lip.layers import (
    SpectralDense,
    SpectralConv2D,
    ScaledL2NormPooling2D,
    FrobeniusDense,
)
from deel.lip.model import Sequential
from deel.lip.activations import GroupSort
from deel.lip.losses import MulticlassHKR, MulticlassKR, HKR, HingeMargin
from keras.layers import Input, Flatten
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
import keras.ops as K
import matplotlib.pyplot as plt
import torchattacks
import torch
import torch.nn as nn
import torchattacks
from robustbench.utils import clean_accuracy
import pandas as pd
import pickle


from radius_evaluation_tools import *
from data_processing import * 


# model = keras.models.load_model("/home/aws_install/robustess_project/deel-lip/docs/notebooks/demo4_vanilla_fashionMNIST_channelfirst.keras")
# model.compile(
#     # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
#     # note also in the case of lipschitz networks, more robustness require more parameters.
#     loss=MulticlassHKR(alpha=100, min_margin=0.25),
#     optimizer=Adam(1e-4),
#     metrics=["accuracy", MulticlassKR()],)
# model.summary()



if __name__ == "__main__":
    # Load Dataset
    print("loading Data :")
    x_train, x_test, y_train, y_test, y_test_ord = load_data("MNIST08")
    # Load Model
    print("Loading Model :")
    # vanilla_model = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo3_FC_vanilla_MNIST08_channelfirst_False_disj_Neurons.keras")
    # vanilla_model_bis = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo3_FC_vanilla_MNIST08_channelfirst_False_disj_Neurons_4logits.keras")

    vanilla_model = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo3_FC_vanilla_MNIST08_channelfirst_False_disj_Neurons_single_output.keras")
    vanilla_model_bis = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo3_FC_vanilla_MNIST08_channelfirst_False_disj_Neurons_single_output_converted_4logits.keras")
    print("Compiling Model :")
    vanilla_model.compile(
   
    loss=HKR(
        alpha=10.0, min_margin=1.0
    ),  # HKR stands for the hinge regularized KR loss
    metrics=[
        # KR,  # shows the KR term of the loss
        HingeMargin(min_margin=1.0),  # shows the hinge term of the loss
    ],
    optimizer=Adam(learning_rate=0.001),)
    vanilla_model_bis.compile(
        # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
        # note also in the case of lipschitz networks, more robustness require more parameters.
        loss=MulticlassHKR(alpha=100, min_margin=0.25),
        optimizer=Adam(1e-4),
        metrics=["accuracy", MulticlassKR()],)
    print("Generating Sample :")
    # Generate the test sample for radius evaluation
    images, labels, idx_list = select_data_for_radius_evaluation_MNIST08(x_test, y_test_ord, vanilla_model_bis)
   
    # total_points = images.shape[0]
    total_points = 10

    # Compute Lipschitz Pessimistic Certificates
    print("Generating Certificates :")
    lip_radius = compute_binary_certificate(images, vanilla_model)
    

    # Initialize the CSV file with column headers
    columns = ["Index", "Label_GT", "Predicted_Label", "Lipschitz_Constant", "Robust_Epsilon", "Adv_Epsilon_AA", "Adv_Epsilon_PGD"]
    csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST08_single_output_10first.csv"
    pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST08_single_output_10first.pkl"

    # Create an empty file with headers
    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

    df_list = []  # Temporary list for storage before Pickle

    for i in range(total_points):
        eps_pgd = single_compute_optimistic_radius_PGD(i, images, labels, lip_radius, vanilla_model_bis, n_iter=10)
        eps_aa = single_compute_optimistic_radius_AA_binary(i,images, labels, lip_radius, vanilla_model_bis, n_iter=10)
        print("Point ", i, "attaques trouvées :", eps_pgd, eps_aa)
        # Create a row
        row = {
            "Index": i ,
            "Label_GT": labels[i].detach().cpu().numpy(),
            "Predicted_Label": np.argmax(vanilla_model(images[i:i+1]).detach().cpu().numpy(), axis=1)[0],
            "Lipschitz_Constant": 1.0,
            "Robust_Epsilon": lip_radius[i].detach().cpu().numpy(),
            "Adv_Epsilon_AA": eps_aa[0].detach().cpu().numpy(),
            "Adv_Epsilon_PGD": eps_pgd[0].detach().cpu().numpy()
        }
        
        # Append to CSV file without rewriting the header
        pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)
        
        # Append to the list for Pickle
        df_list.append(row)
        
        # Save to Pickle at each iteration
        pd.DataFrame(df_list).to_pickle(pkl_path)

