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

# from lipschitz_decomon_tools import get_local_maximum_multiclass, echantillonner_boule_l2_simple, create_difference_model

import sys
sys.path.append('..')

from radius_evaluation_tools import single_compute_relaxation_radius_multiclass


from data_processing import load_data, select_data_for_radius_evaluation
from radius_evaluation_tools import compute_binary_certificate, starting_point_dichotomy

x_train, x_test, y_train, y_test, y_test_ord = load_data("MNIST")

vanilla_model = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo0_vanilla_MNIST_channelfirst_False_disj_Neurons.keras")
vanilla_model.compile(
        # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
        # note also in the case of lipschitz networks, more robustness require more parameters.
        loss=MulticlassHKR(alpha=50, min_margin=0.05),
        optimizer=keras.optimizers.Adam(1e-3),
        metrics=["accuracy", MulticlassKR()],)

images, labels, idx_list = select_data_for_radius_evaluation(x_test, y_test_ord, vanilla_model)

total_points = images.shape[0]

# Initialize the CSV file with column headers
input_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_Decomon_MNIST_single_output_Decomon.csv"
input_pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_Decomon_MNIST_single_output_Decomon.pkl"
output_csv_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_Decomon_MNIST_single_output_Relaxation.csv"
output_pkl_path = "/home/aws_install/robustess_project/lip_notebooks/data/Radius_Data/Radius_Decomon_MNIST_single_output_Relaxation.pkl"

nb_pts = 5

list_eps = []
for i in range(total_points):
    print(i)
    eps_working = single_compute_relaxation_radius_multiclass(i, images, labels, vanilla_model, nb_pts)
    list_eps.append(eps_working)
    break

df = pd.read_csv(input_csv_path)

df["Relaxation"] = list_eps
df.to_pickle(output_pkl_path)
df.to_csv(output_csv_path, mode='a', header=False, index=False)