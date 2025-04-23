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
from deel.lip.losses import MulticlassHKR, MulticlassKR
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




if __name__ == "__main__":
    # Load Dataset
    print("loading Data :")
    x_train, x_test, y_train, y_test, y_test_ord = load_data("MNIST")
    # Load Model
    print("Loading Model :")
    vanilla_model = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo0_vanilla_MNIST_channelfirst.keras")
    print("Compiling Model :")
    vanilla_model.compile(
        # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
        # note also in the case of lipschitz networks, more robustness require more parameters.
        loss=MulticlassHKR(alpha=50, min_margin=0.05),
        optimizer=Adam(1e-3),
        metrics=["accuracy", MulticlassKR()],
    )
    print("Generating Sample :")
    # Generate the test sample for radius evaluation
    images, labels = select_data_for_radius_evaluation(x_test, y_test_ord, vanilla_model)
    # Compute Lipschitz Pessimistic Certificates
    print("Generating Certificates :")
    lip_radius = compute_certificate(images, vanilla_model)
    
    eps_PGD = []
    eps_AA = []

    print('starting PGD : ')
    print('----------------------------------------------------------------------')
    for i in range(images.shape[0]):
        print(i)
        eps_PGD.append(single_compute_optimistic_radius_PGD(images[i:i+1], labels[i:i+1], lip_radius[i:i+1], vanilla_model, n_iter = 10))

    with open("/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/eps_PGD_MNIST.pkl", "wb") as f:
        pickle.dump(eps_PGD, f)

    print('starting AA : ')
    print('----------------------------------------------------------------------')
    for i in range(images.shape[0]):
        print(i)
        eps_AA.append(single_compute_optimistic_radius_AA(images[i:i+1], labels[i:i+1], lip_radius[i:i+1], vanilla_model, n_iter = 10))
    
    with open("/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/eps_AA_MNIST.pkl", "wb") as f:
        pickle.dump(eps_AA, f)

    total_points = images.shape[0]

    # Création du DataFrame avec une colonne d'index de 1 à 200
    df = pd.DataFrame({
        'Index': np.arange(1, total_points + 1),
        'Label_GT': labels.detach().cpu().numpy(),  
        'Label_Predit': np.argmax(vanilla_model(images).detach().cpu().numpy(), axis=1),  
        'Constante_Lipschitz': np.ones(total_points), 
        'Epsilon_Robuste': lip_radius,
        'Epsilon_Adv_AA': eps_AA,
        'Epsilon_Adv_PGD': eps_PGD
    })
    #CSV
    df.to_csv("/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST.csv", index=False)  
    #Pickle
    df.to_pickle("/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST.pkl")

