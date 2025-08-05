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
from keras.layers import Input, Flatten, Dense
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from deel import torchlip
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

from abstract_interpretation_tools import *

def load_MNIST08():
    print("--loading model : --")
    vanilla_model = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo3_FC_vanilla_MNIST08_channelfirst_False_disj_Neurons_single_output.keras")
    vanilla_model.compile(
    
        loss=HKR(
            alpha=10.0, min_margin=1.0
        ),  # HKR stands for the hinge regularized KR loss
        metrics=[
            # KR,  # shows the KR term of the loss
            HingeMargin(min_margin=1.0),  # shows the hinge term of the loss
        ],
        optimizer=Adam(learning_rate=0.001),)
    vanilla_model.summary()

    print("--create torch model : --")
    pytorch_model = torchlip.Sequential(
        nn.Flatten(),
        torchlip.SpectralLinear(in_features=784, out_features=32),
        MaxMin(),
        torchlip.SpectralLinear(in_features=32, out_features=16),
        MaxMin(),
        torchlip.SpectralLinear(in_features=16, out_features=1),
    )
    return pytorch_model, vanilla_model

def load_MNIST():
    print("--loading model : --")
    vanilla_model = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo0_vanilla_MNIST_channelfirst_False_disj_Neurons.keras")
    vanilla_model.compile(
        # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
        # note also in the case of lipschitz networks, more robustness require more parameters.
        loss=MulticlassHKR(alpha=50, min_margin=0.05),
        optimizer=Adam(1e-3),
        metrics=["accuracy", MulticlassKR()],)
    vanilla_model.summary()

    print("--create torch model : --")
    pytorch_model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding="same"),
        MaxMin(),
        ScaledL2NormPool2d((2,2)),
        torchlip.SpectralConv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding="same"),
        MaxMin(),
        ScaledL2NormPool2d((2,2)),
        nn.Flatten(),
        torchlip.SpectralLinear(784, 32),
        MaxMin(),
        torchlip.SpectralLinear(32,10, bias=False),
    )
    return pytorch_model, vanilla_model


if __name__ == "__main__":

    pytorch_model, vanilla_model = load_MNIST()
    # Set to evaluation mode
    pytorch_model = pytorch_model.vanilla_export()
    pytorch_model.eval()
    
    pytorch_model = convert_weights_dynamically_cnn(vanilla_model, pytorch_model)

    # kr_model = keras.Sequential([
    # keras.layers.Input(shape=(1, 28, 28)),
    # GroupSort(2)
    # ])
    # pt_model = nn.Sequential(MaxMin())

    test_vec = torch.rand((1,28,28))[None]
    # print(pt_model(test_vec), kr_model(test_vec))


    print(pytorch_model(test_vec), vanilla_model(test_vec))

    check_first_conv_weights(vanilla_model, pytorch_model)

    