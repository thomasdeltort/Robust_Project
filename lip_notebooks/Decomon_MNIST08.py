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
import numpy as np
import keras.ops as K
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import pickle


from radius_evaluation_tools import *
from data_processing import * 

from decomon.layers import DecomonLayer
from decomon.models import clone
from lipschitz_custom_tools import affine_bound_groupsort_output_keras, affine_bound_sqrt_output_keras, affine_bound_square_output_keras
from decomon.perturbation_domain import BallDomain
from decomon import get_lower_noise, get_range_noise, get_upper_noise


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
   
    total_points = images.shape[0]

    # Initialize the CSV file with column headers
    input_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST08_single_output.csv"
    input_pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST08_single_output.pkl"
    output_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_Decomon_MNIST08_single_output_Decomon.csv"
    output_pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_Decomon_MNIST08_single_output_Decomon.pkl"

    # input_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST_10first.csv"
    # input_pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST_10first.pkl"
    # output_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST_10first_Decomon.csv"
    # output_pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_MNIST_10first_Decomon.pkl"

    list_eps = []

    for i in range(total_points):
        list_eps.append(single_compute_decomon_radius(i, images, labels, vanilla_model).cpu().detach().numpy())
        
    df = pd.read_csv(input_csv_path)

    df["Decomon"] = list_eps
    df.to_pickle(output_pkl_path)
    df.to_csv(output_csv_path, mode='a', header=False, index=False)