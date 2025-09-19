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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

    pytorch_model = pytorch_model.vanilla_export()
    pytorch_model.eval()

    pytorch_model = convert_weights_dynamically_dense(vanilla_model, pytorch_model)

    return pytorch_model

def load_MNIST08_2logits():
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
        torchlip.SpectralLinear(in_features=16, out_features=2),
    )

    pytorch_model = pytorch_model.vanilla_export()
    pytorch_model.eval()

    pytorch_model = convert_weights_dynamically_dense(vanilla_model, pytorch_model)

    return pytorch_model

def load_FMNIST():
    print("--loading model : --")
    vanilla_model = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo4_vanilla_fashionMNIST_channelfirst_False_disj_Neurons.keras")
    # vanilla_model.compile(
    #     # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
    #     # note also in the case of lipschitz networks, more robustness require more parameters.
    #     loss=MulticlassHKR(alpha=100, min_margin=0.25),
    #     optimizer=Adam(1e-4),
    #     metrics=["accuracy", MulticlassKR()],)

    print("--create torch model : --")
    pytorch_model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding="same"),
        MaxMin(),
        ScaledL2NormPool2d((2,2)),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same"),
        MaxMin(),
        ScaledL2NormPool2d((2,2)),
        FlattenChannelLast(),
        torchlip.SpectralLinear(1568, 64),
        MaxMin(),
        torchlip.SpectralLinear(64,10, bias=False),
    )
    pytorch_model = pytorch_model.vanilla_export()
    pytorch_model.eval()

    pytorch_model = convert_weights_dynamically_cnn(vanilla_model, pytorch_model)

    return pytorch_model


def load_MNIST(evaluation=False):
    print("--loading model : --")
    vanilla_model = keras.models.load_model("/home/aws_install/robustess_project/lip_models/demo0_vanilla_MNIST_channelfirst_False_disj_Neurons.keras")
    # vanilla_model.compile(
    #     # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
    #     # note also in the case of lipschitz networks, more robustness require more parameters.
    #     loss=MulticlassHKR(alpha=50, min_margin=0.05),
    #     optimizer=Adam(1e-3),
    #     metrics=["accuracy", MulticlassKR()],)
    vanilla_model.summary()

    print("--create torch model : --")
    pytorch_model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding="same"),
        MaxMin(),
        ScaledL2NormPool2d((2,2), stride=2),
        # torchlip.modules.ScaledL2NormPool2d((2,2)),
        torchlip.SpectralConv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding="same"),
        MaxMin(),
        ScaledL2NormPool2d((2,2), stride=2),
        # torchlip.modules.ScaledL2NormPool2d((2,2)),
        FlattenChannelLast(),
        torchlip.SpectralLinear(784, 32),
        MaxMin(),
        torchlip.SpectralLinear(32,10, bias=False),
    )
    pytorch_model = pytorch_model.vanilla_export()
    pytorch_model.eval()

    pytorch_model = convert_weights_dynamically_cnn(vanilla_model, pytorch_model)

    if evaluation :
        test_vec = torch.ones((1,28,28))[None]

        # debug_and_compare_submodels(vanilla_model_unfolded, pytorch_model, test_vec)

        print(pytorch_model(test_vec), vanilla_model(test_vec))
    return pytorch_model



if __name__ == "__main__":

    
    pytorch_model = load_MNIST()
    torch.save(pytorch_model.state_dict(), '/home/aws_install/robustess_project/lip_models/model_MNIST.pt')

    pytorch_model = load_FMNIST()
    torch.save(pytorch_model.state_dict(), '/home/aws_install/robustess_project/lip_models/model_FMNIST.pt')

    pytorch_model = load_MNIST08()
    torch.save(pytorch_model.state_dict(), '/home/aws_install/robustess_project/lip_models/model_MNIST08.pt')

    # # vanilla_model_unfolded = unfold_keras_model(vanilla_model)

    # # vanilla_model_unfolded.summary()
    # # Set to evaluation mode
    # pytorch_model = pytorch_model.vanilla_export()
    # pytorch_model.eval()
    
    # pytorch_model = convert_weights_dynamically_cnn(vanilla_model, pytorch_model)

   
    

    # test_vec = torch.ones((1,28,28))[None]

    # # debug_and_compare_submodels(vanilla_model_unfolded, pytorch_model, test_vec)

    # print(pytorch_model(test_vec), vanilla_model(test_vec))

    # if eval_test:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])

    #     # --- 2. Load Data ---

    #     # Download and load the MNIST test dataset
    #     test_dataset = datasets.MNIST(
    #         root='./data', 
    #         train=False, 
    #         download=True,
    #         transform=transform
    #     )

    #     # Create a DataLoader to handle batching of the test data
    #     test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    #     accuracy = evaluate_model(pytorch_model, device, test_loader)

    #     print(accuracy)