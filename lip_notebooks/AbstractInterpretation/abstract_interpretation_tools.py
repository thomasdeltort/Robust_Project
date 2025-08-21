import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
# from deel.lip.layers import (
#     SpectralDense,
#     SpectralConv2D,
#     ScaledL2NormPooling2D,
#     FrobeniusDense,
# )
# from deel.lip.model import Sequential
# from deel.lip.activations import GroupSort
# from deel.lip.losses import MulticlassHKR, MulticlassKR, HKR, HingeMargin
from keras.layers import Input, Flatten, Dense, Conv2D
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from deel import torchlip
import numpy as np
import keras.ops as K
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MaxMin(nn.Module):
    """
    Custom activation layer that sorts features in pairs.
    Equivalent to torchlip.GroupSort2() but compatible with auto_LiRPA.
    """
    def forward(self, x):
        # Ensure the last dimension has an even number of features
        # if x.shape[-1] % 2 != 0:
        #     raise ValueError("The last dimension must be even for MaxMin activation.")

        # Reshape the tensor to group the last dimension into pairs
        # (batch_size, ..., num_features) -> (batch_size, ..., num_features/2, 2)
        x_pairs = x.view(*x.shape[:-1], -1, 2)

        # Separate the pairs and compute min and max
        a = x_pairs[..., 0]
        b = x_pairs[..., 1]
        min_vals = torch.min(a, b)
        max_vals = torch.max(a, b)

        # Stack them back together to form sorted pairs
        # The result has min followed by max for each pair
        sorted_pairs = torch.stack((min_vals, max_vals), dim=-1)

        # Reshape back to the original input shape
        return sorted_pairs.view(x.shape)

class ReLU_GroupSort2(nn.Module):
    """
    A PyTorch module that sorts pairs of features (groups of 2) along the last
    dimension. This implementation is a reformulation of the GroupSort2 activation
    function using only dense (Linear) layers and ReLU activations, making it
    compatible with network verification tools like autolirpa.

    The input tensor's total number of features (product of dimensions after
    the batch dimension) must be even.
    """
    def __init__(self):
        super(ReLU_GroupSort2, self).__init__()
        
        # Layer 1: Computes x2 - x1 for a pair [x1, x2]
        # Keras kernel shape: (in_features, out_features) = (2, 1) -> [[-1.0], [1.0]]
        # PyTorch weight shape: (out_features, in_features) = (1, 2)
        self.layer1 = nn.Linear(2, 1, bias=False)
        w1 = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)
        self.layer1.weight = nn.Parameter(w1, requires_grad=False)

        self.relu = nn.ReLU()

        # Layer 3: Takes relu(x2-x1) and produces [-relu(x2-x1), relu(x2-x1)]
        # Keras kernel shape: (in_features, out_features) = (1, 2) -> [[-1.0, 1.0]]
        # PyTorch weight shape: (out_features, in_features) = (2, 1)
        self.layer3 = nn.Linear(1, 2, bias=False)
        w3 = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
        self.layer3.weight = nn.Parameter(w3, requires_grad=False)

        # Layer 4: Permutes an input pair [x1, x2] to [x2, x1]
        # Keras kernel shape: (in_features, out_features) = (2, 2) -> [[0,1],[1,0]]
        # PyTorch weight shape: (out_features, in_features) = (2, 2)
        # The permutation matrix is symmetric, so it's the same.
        self.layer4 = nn.Linear(2, 2, bias=False)
        w4 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        self.layer4.weight = nn.Parameter(w4, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original shape to reshape at the end
        original_shape = x.shape
        batch_size = original_shape[0]
        
        # Check if the total number of features is even
        num_features = np.prod(original_shape[1:])
        if num_features % 2 != 0:
            raise ValueError(f"The total number of features must be even, but got {num_features}.")

        # Reshape the input into groups of 2. Shape: (batch_size, n_dim, 2)
        # where n_dim is the total number of pairs.
        # reshaped_x = x.view(original_shape[0], -1, 2)
        reshaped_x = x.reshape(batch_size, -1, 2)

        # Step 1: Compute x2 - x1
        # output_1 shape: (batch_size, n_dim, 1)
        output_1 = self.layer1(reshaped_x)

        # Step 2: Apply ReLU -> relu(x2 - x1)
        # output_2 shape: (batch_size, n_dim, 1)
        output_2 = self.relu(output_1)

        # Step 3: Create [-relu(x2-x1), relu(x2-x1)]
        # output_3 shape: (batch_size, n_dim, 2)
        output_3 = self.layer3(output_2)

        # Step 4: Permute the original input pair to [x2, x1]
        # output_4 shape: (batch_size, n_dim, 2)
        output_4 = self.layer4(reshaped_x)

        # Step 5: Add the two results to get the sorted pair
        # [-relu(x2-x1)+x2, relu(x2-x1)+x1] = [min(x1, x2), max(x1, x2)]
        sorted_pairs = output_3 + output_4

        # Step 6: Reshape the sorted pairs back to the original input shape
        output = sorted_pairs.view(original_shape)

        return output


def convert_weights_dynamically_dense(keras_model, pytorch_model):
    """
    Copies weights by looping through corresponding layers.
    This function is robust and handles PyTorch models both with and without spectral_norm.
    """
    print("Starting dynamic weight conversion...")
    
    # 1. Get the layers that have weights to copy
    keras_dense_layers = [l for l in keras_model.layers if isinstance(l, Dense)]
    
    # We get all linear layers from the PyTorch model. nn.Sequential helps here.
    pytorch_linear_modules = [m for m in pytorch_model.modules() if isinstance(m, nn.Linear)]

    
    if len(keras_dense_layers) != len(pytorch_linear_modules):
        raise ValueError("Model architectures do not match! "
                         f"Found {len(keras_dense_layers)} SpectralDense layers in Keras "
                         f"and {len(pytorch_linear_modules)} Linear layers in PyTorch.")

    # 2. Loop through the layer pairs
    for i, (k_layer, pt_module) in enumerate(zip(keras_dense_layers, pytorch_linear_modules)):
        print(f"\n--- Processing Layer {i+1}: Keras '{k_layer.name}' -> PyTorch ---")
        
        k_weights = k_layer.get_weights()
        
        if k_layer.use_bias:
            k_w, k_b = k_weights 
            pt_module.bias.data.copy_(torch.from_numpy(k_b))
        else:
            k_w = k_weights[0] 
        # import pdb ; pdb.set_trace()
            
        pt_module.weight.data.copy_(torch.from_numpy(k_w.T))
            
    print("\n\nDynamic weight conversion finished.")
    return pytorch_model

def convert_weights_dynamically_cnn(keras_model, pytorch_model):
    """
    Copies weights from a Keras model to a PyTorch model, supporting both 
    Dense/Linear and Conv2D/Conv2d layers.

    This function dynamically identifies corresponding layers and correctly
    transposes the weight tensors for compatibility.
    """
    print("Starting dynamic weight conversion for Dense and Conv2D layers...")

    # 1. Get the layers that have weights to copy from both models
    # We filter for the specific layer types we know how to handle.
    keras_layers = [l for l in keras_model.layers if isinstance(l, (Dense, Conv2D))]
    pytorch_modules = [m for m in pytorch_model.children() if isinstance(m, (nn.Linear, nn.Conv2d))]

    # 2. Sanity check to ensure architectures are compatible
    if len(keras_layers) != len(pytorch_modules):
        raise ValueError("Model architectures do not match! "
                         f"Found {len(keras_layers)} trainable layers (Dense, Conv2D) in Keras "
                         f"and {len(pytorch_modules)} trainable modules (Linear, Conv2d) in PyTorch.")

    # 3. Loop through the corresponding layer pairs
    for i, (k_layer, pt_module) in enumerate(zip(keras_layers, pytorch_modules)):
        print(f"\n--- Processing Layer {i+1}: Keras '{k_layer.name}' ({type(k_layer).__name__}) -> PyTorch ({type(pt_module).__name__}) ---")

        # Get the weights from the Keras layer
        k_weights = k_layer.get_weights()

        # --- Case 1: Handle Dense -> Linear layers ---
        if isinstance(k_layer, Dense) and isinstance(pt_module, nn.Linear):
            if k_layer.use_bias:
                k_w, k_b = k_weights
                # Copy bias
                pt_module.bias.data.copy_(torch.from_numpy(k_b))
            else:
                k_w = k_weights[0]
            
            # Keras Dense kernel shape: (in_features, out_features)
            # PyTorch Linear weight shape: (out_features, in_features)
            # We need to transpose the Keras weights.
            pt_module.weight.data.copy_(torch.from_numpy(k_w.T))
            # import pdb ; pdb.set_trace()
        # --- Case 2: Handle Conv2D -> Conv2d layers ---
        elif isinstance(k_layer, Conv2D) and isinstance(pt_module, nn.Conv2d):
            if k_layer.use_bias:
                k_w, k_b = k_weights
                # Bias shapes are compatible: (out_channels,)
                pt_module.bias.data.copy_(torch.from_numpy(k_b))
            else:
                k_w = k_weights[0]
            
            # We need to permute the axes from (0, 1, 2, 3) to (3, 2, 0, 1).
            k_w_transposed = np.transpose(k_w, (3, 2, 0, 1))

            ### TODO PROBLEM HERE
            pt_module.weight.data.copy_(torch.from_numpy(k_w_transposed))
            # import pdb ; pdb.set_trace()
            ### HERE

        # --- Case 3: Handle mismatched layer types ---
        else:
            raise TypeError(f"Layer type mismatch at index {i}: "
                            f"Keras layer is {type(k_layer).__name__} but PyTorch module is {type(pt_module).__name__}.")

    print("\n\nDynamic weight conversion finished successfully.")
    return pytorch_model

def load_MNIST08():
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
    # Load the saved weights into the model
    pytorch_model.load_state_dict(torch.load('/home/aws_install/robustess_project/lip_models/model_MNIST08.pt'))
    pytorch_model.eval()
    return pytorch_model

def load_FMNIST():
    print("--create torch model : --")
    pytorch_model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
        MaxMin(),
        ScaledL2NormPool2d((2,2)),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
        MaxMin(),
        ScaledL2NormPool2d((2,2)),
        FlattenChannelLast(),
        torchlip.SpectralLinear(1568, 64),
        MaxMin(),
        torchlip.SpectralLinear(64,10, bias=False),
    )
    pytorch_model = pytorch_model.vanilla_export()
    # Load the saved weights into the model
    pytorch_model.load_state_dict(torch.load('/home/aws_install/robustess_project/lip_models/model_FMNIST.pt'))
    pytorch_model.eval()
    return pytorch_model


def load_MNIST():
    print("--create torch model : --")
    pytorch_model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
        MaxMin(),
        ScaledL2NormPool2d((2,2), stride=2),
        # torchlip.modules.ScaledL2NormPool2d((2,2)),
        torchlip.SpectralConv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1),
        MaxMin(),
        ScaledL2NormPool2d((2,2), stride=2),
        # torchlip.modules.ScaledL2NormPool2d((2,2)),
        FlattenChannelLast(),
        torchlip.SpectralLinear(784, 32),
        MaxMin(),
        torchlip.SpectralLinear(32,10, bias=False),
    )
    pytorch_model = pytorch_model.vanilla_export()
    # Load the saved weights into the model
    pytorch_model.load_state_dict(torch.load('/home/aws_install/robustess_project/lip_models/model_MNIST.pt'))
    pytorch_model.eval()
    return pytorch_model


def inspect_first_layer_weights(keras_model, pytorch_model):
    """
    Prints the weights and biases of the first dense layer for both models
    to help with verification and debugging.
    """
    print("\n=================================================")
    print("--- Inspecting First Dense Layer Weights ---")
    print("=================================================")

    # --- 1. Inspect Keras Model ---
    print("\n--- Keras Model (spectral_dense) ---")
    try:
        keras_first_dense = keras_model.get_layer('spectral_conv2d')
        k_weights = keras_first_dense.get_weights()
        
        # In deel-lip, weights are [kernel, bias, u_vector]
        k_kernel, k_bias = k_weights
        
        print(f"Kernel shape: {k_kernel.shape} (Inputs, Outputs)")
        print("Kernel (top-left 4x4 slice):\n", k_kernel[:4, :4])
        
        print(f"\nBias shape: {k_bias.shape}")
        print("Bias (first 4 values):\n", k_bias[:4])

    except Exception as e:
        print(f"Could not inspect Keras model. Error: {e}")


    # --- 2. Inspect PyTorch Model ---
    print("\n--- PyTorch Model (first nn.Linear module) ---")
    try:
        # Find the first nn.Linear module in the PyTorch model
        pt_first_linear = next(m for m in pytorch_model.modules() if isinstance(m, nn.Conv2d))
        
    
        print("Layer is a standard nn.Linear (no spectral_norm).")
        # Access the standard weight parameter
        pt_weight = pt_first_linear.weight.data

        pt_bias = pt_first_linear.bias.data
        
        # NOTE: PyTorch stores weights as (Outputs, Inputs), so it's the transpose of Keras's kernel
        print(f"Weight shape: {pt_weight.shape} (Outputs, Inputs) -> Note the transpose!")
        print("Weight (top-left 4x4 slice):\n", pt_weight[:4, :4])

        print(f"\nBias shape: {pt_bias.shape}")
        print("Bias (first 4 values):\n", pt_bias[:4])
        
        print("\nReminder: Keras kernel should be the transpose of PyTorch weight.")

    except Exception as e:
        print(f"Could not inspect PyTorch model. Error: {e}")

def check_first_conv_weights(keras_model, pytorch_model):
    """
    Compares the weights of the first Conv2D layer in a Keras model against
    the first Conv2d layer in a PyTorch model.

    Args:
        keras_model (keras.Model): The Keras model.
        pytorch_model (torch.nn.Module): The PyTorch model.

    Returns:
        bool: True if weights and biases are identical (within a tolerance), 
              False otherwise.
    """
    print("\n--- Verifying weights of the first convolutional layer ---")
    
    # 1. Find the first convolutional layer in each model
    first_keras_conv = None
    for layer in keras_model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            first_keras_conv = layer
            break

    first_pytorch_conv = None
    for module in pytorch_model.modules():
        if isinstance(module, nn.Conv2d):
            first_pytorch_conv = module
            break

    # 2. Handle cases where a conv layer isn't found
    if not first_keras_conv:
        print("❌ Error: No Conv2D layer found in the Keras model.")
        return False
    if not first_pytorch_conv:
        print("❌ Error: No Conv2d layer found in the PyTorch model.")
        return False
        
    print(f"Found Keras layer: '{first_keras_conv.name}'")
    print(f"Found PyTorch module: {first_pytorch_conv}")

    # 3. Extract Keras weights
    k_weights = first_keras_conv.get_weights()
    # k_kernel = np.transpose(k_weights[0], (3, 2, 0, 1))
    k_kernel = k_weights[0]
    k_bias = k_weights[1] if first_keras_conv.use_bias else None

    # 4. Extract PyTorch weights and convert to NumPy
    pt_kernel = first_pytorch_conv.weight.data.detach().numpy()
    pt_bias = first_pytorch_conv.bias.data.detach().numpy() if first_pytorch_conv.bias is not None else None
    
    # 5. IMPORTANT: Transpose Keras kernel if it's in 'channels_last' format
    print(f"Keras layer data_format: '{first_keras_conv.data_format}'")
    # data_format = first_keras_conv.data_format
    data_format = 'channels_last'
    if  data_format == 'channels_last':
        # Keras (H, W, C_in, C_out) -> PyTorch (C_out, C_in, H, W)
        print("Transposing Keras kernel from (H, W, C_in, C_out) to match PyTorch...")
        k_kernel = np.transpose(k_kernel, (3, 2, 1, 0))

    # 6. Compare shapes
    print(f"Keras Kernel Shape (post-transpose): {k_kernel.shape}")
    print(f"PyTorch Kernel Shape:                {pt_kernel.shape}")
    
    if k_kernel.shape != pt_kernel.shape:
        print("❌ FAILURE: Kernel shapes do not match.")
        # return False

    # 7. Compare the weights and biases using numpy.allclose for float safety
    kernels_match = np.allclose(k_kernel, pt_kernel, atol=1e-6)
    if kernels_match:
        print("✅ SUCCESS: Kernel weights are identical.")
    else:
        print("❌ FAILURE: Kernel weights are different.")
        print(f"Max absolute difference: {np.abs(k_kernel - pt_kernel).max()}")

    biases_match = True # Assume true if no bias
    if k_bias is not None and pt_bias is not None:
        biases_match = np.allclose(k_bias, pt_bias, atol=1e-6)
        if biases_match:
            print("✅ SUCCESS: Bias weights are identical.")
        else:
            print("❌ FAILURE: Bias weights are different.")
            print(f"Max absolute difference: {np.abs(k_bias - pt_bias).max()}")
    
    return kernels_match and biases_match


# from torch.nn.modules.utils import _pair
# from typing import Optional, Union
# from torch.nn.common_types import _size_2_t

# # --- Helper Module for a LiRPA-compatible export ---
# # This module encapsulates the LiRPA-compatible operations so that the exported
# # model does not depend on our custom ScaledL2NormPool2d class definition.
# class _ExportedL2Pool(nn.Module):
#     def __init__(self, kernel_size, stride, ceil_mode, coeff):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.ceil_mode = ceil_mode
#         self.coeff = coeff
#         self.num_elements = self.kernel_size[0] * self.kernel_size[1]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_squared = torch.pow(x, 2)
#         avg_of_squares = F.avg_pool2d(
#             x_squared,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             ceil_mode=self.ceil_mode
#         )
#         sum_of_squares = avg_of_squares * self.num_elements
#         pooled = torch.sqrt(sum_of_squares + 1e-9)
#         return pooled * self.coeff

#     def __repr__(self):
#         return (f"_ExportedL2Pool(kernel_size={self.kernel_size}, "
#                 f"stride={self.stride}, coeff={self.coeff})")
    
# def computePoolScalingFactor(kernel_size):
#     if isinstance(kernel_size, tuple):
#         scalingFactor = math.sqrt(np.prod(np.asarray(kernel_size)))
#     else:
#         scalingFactor = kernel_size
#     return scalingFactor

# class ScaledL2NormPool2d(torch.nn.Module, torchlip.module.LipschitzModule):
#     def __init__(
#         self,
#         kernel_size: _size_2_t,
#         stride: Optional[_size_2_t] = None,
#         ceil_mode: bool = False,
#         k_coef_lip: float = 1.0,
#     ):
#         """
#         auto_LiRPA-compatible L2-norm pooling layer.
#         """
#         # We no longer inherit from LPPool2d, but directly from our custom base class
#         # and nn.Module (via LipschitzModule).
#         torch.nn.Module.__init__(self)
#         torchlip.module.LipschitzModule.__init__(self, k_coef_lip)
        
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride) if stride is not None else self.kernel_size
#         self.ceil_mode = ceil_mode

#         self.scalingFactor = computePoolScalingFactor(self.kernel_size)

#         if self.stride != self.kernel_size:
#             raise RuntimeError("For provable robustness, stride must be equal to kernel_size for this implementation.")

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 1. Square the input tensor element-wise.
#         # This is a basic operation that auto_LiRPA can handle.
#         x_squared = torch.pow(x, 2)
        
#         # 2. Apply average pooling.
#         # auto_LiRPA has native support for AvgPool2d.
#         sum_squared = F.avg_pool2d(
#             x_squared,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             ceil_mode=self.ceil_mode
#         )
        
#         # 3. Get the number of elements in the pooling window.
#         num_elements_in_kernel = self.kernel_size[0] * self.kernel_size[1]
        
#         # avg_pool(x^2) = (sum(x^2)) / N  =>  sum(x^2) = avg_pool(x^2) * N
#         sum_squared = sum_squared * num_elements_in_kernel
        
#         # 4. Take the element-wise square root.
#         # torch.sqrt is also a standard supported operation.
#         # Adding a small epsilon for numerical stability to avoid sqrt(0) gradients issues.
#         pooled = torch.sqrt(sum_squared + 1e-8)
        
#         # 5. Apply the Lipschitz scaling factor.
#         return pooled * self._coefficient_lip 
#     # * self.scalingFactor
        
#     def __repr__(self):
#         return (f"ScaledL2NormPool2d(kernel_size={self.kernel_size}, "
#                 f"stride={self.stride}, k_coef_lip={self._coefficient_lip})")
    
#     def vanilla_export(self) -> nn.Module:
#         """
#         Exports the layer to a self-contained, auto_LiRPA-compatible nn.Module.

#         This function returns a new module that encapsulates the exact same
#         LiRPA-compatible operations as this layer's forward pass. This is
#         somewhat redundant, as this layer itself is already compatible.
#         The primary use for this would be to create a model with no custom
#         class definitions before saving or deployment.

#         IMPORTANT: For LiRPA analysis, you can use the main ScaledL2NormPool2d
#         layer directly. You do not need to call this export function first.
#         """
#         # This returns a new, standard nn.Module that is also LiRPA-compatible.
#         return _ExportedL2Pool(
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             ceil_mode=self.ceil_mode,
#             coeff=self._coefficient_lip
#         )

# class _ExportedAdaptiveL2Pool(nn.Module):
#     def __init__(self, output_size, coeff):
#         super().__init__()
#         self.output_size = output_size
#         self.coeff = coeff
#         self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Get spatial dimensions to calculate total number of elements
#         h, w = x.shape[-2:]
#         num_elements = h * w

#         # LiRPA-compatible L2 norm calculation
#         x_squared = torch.pow(x, 2)
#         # adaptive_avg_pool computes sum(x^2) / num_elements
#         avg_of_squares = self.adaptive_avg_pool(x_squared)
#         sum_of_squares = avg_of_squares * num_elements
#         pooled = torch.sqrt(sum_of_squares + 1e-9)
        
#         return pooled * self.coeff

#     def __repr__(self):
#         return (f"_ExportedAdaptiveL2Pool(output_size={self.output_size}, "
#                 f"coeff={self.coeff})")
    
# class ScaledAdaptiveL2NormPool2d(torch.nn.Module, torchlip.module.LipschitzModule):
#     def __init__(
#         self,
#         output_size: _size_2_t = (1, 1),
#         k_coef_lip: float = 1.0,
#     ):
#         """
#         auto_LiRPA-compatible Adaptive L2-norm pooling layer.

#         This layer's forward pass is implemented using only operations natively
#         supported by auto_LiRPA (pow, adaptive_avg_pool2d, sqrt, mul).
#         """
#         torch.nn.Module.__init__(self)
#         torchlip.module.LipschitzModule.__init__(self, k_coef_lip)
        
#         # Ensure output_size is a tuple of two integers
#         if not isinstance(output_size, tuple) or len(output_size) != 2:
#              output_size = _pair(output_size)

#         # For this operation to be a valid norm, it must collapse the spatial dimensions
#         if output_size[0] != 1 or output_size[1] != 1:
#             raise ValueError("output_size must be (1, 1) for ScaledAdaptiveL2NormPool2d")
        
#         self.output_size = output_size
#         # We use the standard AdaptiveAvgPool2d as a supported building block
#         self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(self.output_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Performs adaptive L2 pooling using a sequence of LiRPA-compatible operations.
#         """
#         # 1. Get spatial dimensions to calculate total number of elements.
#         h, w = x.shape[-2:]
#         num_elements = h * w

#         # 2. Square the input tensor.
#         x_squared = torch.pow(x, 2)
        
#         # 3. Apply adaptive average pooling to the squared tensor.
#         # This computes (sum of squares) / num_elements
#         avg_of_squares = self.adaptive_avg_pool(x_squared)
        
#         # 4. Multiply by num_elements to get the sum of squares.
#         sum_of_squares = avg_of_squares * num_elements
        
#         # 5. Take the square root to get the L2 norm over the spatial dimensions.
#         # Add a small epsilon for numerical stability.
#         pooled = torch.sqrt(sum_of_squares + 1e-9)
        
#         # 6. Apply the Lipschitz scaling factor.
#         return pooled * self._coefficient_lip
        
#     def __repr__(self):
#         return (f"ScaledAdaptiveL2NormPool2d(output_size={self.output_size}, "
#                 f"k_coef_lip={self._coefficient_lip})")

#     def vanilla_export(self) -> nn.Module:
#         """
#         Exports the layer to a self-contained, auto_LiRPA-compatible nn.Module.
#         """
#         return _ExportedAdaptiveL2Pool(
#             output_size=self.output_size,
#             coeff=self._coefficient_lip
        # )
class _ExportedAdaptiveL2Pool(nn.Module):
    def __init__(self, output_size, coeff):
        super().__init__()
        self.output_size = output_size
        self.coeff = coeff
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get spatial dimensions to calculate total number of elements
        h, w = x.shape[-2:]
        num_elements = h * w

        # LiRPA-compatible L2 norm calculation
        x_squared = torch.pow(x, 2)
        # adaptive_avg_pool computes sum(x^2) / num_elements
        avg_of_squares = self.adaptive_avg_pool(x_squared)
        # Convert the dynamic scalar `num_elements` to a 4D tensor.
        num_elements_4d = torch.tensor(
            num_elements, device=x.device, dtype=x.dtype
        ).view(1, 1, 1, 1)
        sum_of_squares = avg_of_squares * num_elements_4d
        pooled = torch.sqrt(sum_of_squares + 1e-9)
        # Convert the scalar `coeff` to a 4D tensor.
        coeff_4d = torch.tensor(
            self.coeff, device=x.device, dtype=x.dtype
        ).view(1, 1, 1, 1)

        return pooled * coeff_4d

    def __repr__(self):
        return (f"_ExportedAdaptiveL2Pool(output_size={self.output_size}, "
                f"coeff={self.coeff})")
    
from torch.nn.modules.utils import _pair
from typing import Optional, Union
from torch.nn.common_types import _size_2_t

# --- Helper Module for a LiRPA-compatible export ---
# This module encapsulates the LiRPA-compatible operations so that the exported
# model does not depend on our custom ScaledL2NormPool2d class definition.
class _ExportedL2Pool(nn.Module):
    def __init__(self, kernel_size, stride, ceil_mode, coeff):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode
        self.coeff = coeff
        self.num_elements = self.kernel_size[0] * self.kernel_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_squared = torch.pow(x, 2)
        avg_of_squares = F.avg_pool2d(
            x_squared,
            kernel_size=self.kernel_size,
            stride=self.stride,
            ceil_mode=self.ceil_mode
        )
        # --- FIX #1 ---
        # Convert the scalar `num_elements` to a 4D tensor before multiplying.
        num_elements_4d = torch.tensor(
            self.num_elements, device=x.device, dtype=x.dtype
        ).view(1, 1, 1, 1)
        sum_of_squares = avg_of_squares * num_elements_4d
        pooled = torch.sqrt(sum_of_squares + 1e-9)
        # --- FIX #2 ---
        # Convert the scalar `coeff` to a 4D tensor before multiplying.
        coeff_4d = torch.tensor(
            self.coeff, device=x.device, dtype=x.dtype
        ).view(1, 1, 1, 1)
        return pooled * coeff_4d

    def __repr__(self):
        return (f"_ExportedL2Pool(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, coeff={self.coeff})")
    
def computePoolScalingFactor(kernel_size):
    if isinstance(kernel_size, tuple):
        scalingFactor = math.sqrt(np.prod(np.asarray(kernel_size)))
    else:
        scalingFactor = kernel_size
    return scalingFactor

class ScaledL2NormPool2d(torch.nn.Module, torchlip.module.LipschitzModule):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        ceil_mode: bool = False,
        k_coef_lip: float = 1.0,
    ):
        """
        auto_LiRPA-compatible L2-norm pooling layer.
        """
        # We no longer inherit from LPPool2d, but directly from our custom base class
        # and nn.Module (via LipschitzModule).
        torch.nn.Module.__init__(self)
        torchlip.module.LipschitzModule.__init__(self, k_coef_lip)
        
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.ceil_mode = ceil_mode

        self.scalingFactor = computePoolScalingFactor(self.kernel_size)

        if self.stride != self.kernel_size:
            raise RuntimeError("For provable robustness, stride must be equal to kernel_size for this implementation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Square the input tensor element-wise.
        # This is a basic operation that auto_LiRPA can handle.
        x_squared = torch.pow(x, 2)
        
        # 2. Apply average pooling.
        # auto_LiRPA has native support for AvgPool2d.
        sum_squared = F.avg_pool2d(
            x_squared,
            kernel_size=self.kernel_size,
            stride=self.stride,
            ceil_mode=self.ceil_mode
        )
        
        # 3. Get the number of elements in the pooling window.
        num_elements_in_kernel = self.kernel_size[0] * self.kernel_size[1]
        
        # avg_pool(x^2) = (sum(x^2)) / N  =>  sum(x^2) = avg_pool(x^2) * N
        sum_squared = sum_squared * num_elements_in_kernel
        
        # 4. Take the element-wise square root.
        # torch.sqrt is also a standard supported operation.
        # Adding a small epsilon for numerical stability to avoid sqrt(0) gradients issues.
        pooled = torch.sqrt(sum_squared + 1e-8)
        
        # 5. Apply the Lipschitz scaling factor.
        return pooled * self._coefficient_lip 
    # * self.scalingFactor
        
    def __repr__(self):
        return (f"ScaledL2NormPool2d(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, k_coef_lip={self._coefficient_lip})")

    
    def vanilla_export(self) -> nn.Module:
        """
        Exports the layer to a self-contained, auto_LiRPA-compatible nn.Module.

        This function returns a new module that encapsulates the exact same
        LiRPA-compatible operations as this layer's forward pass. This is
        somewhat redundant, as this layer itself is already compatible.
        The primary use for this would be to create a model with no custom
        class definitions before saving or deployment.

        IMPORTANT: For LiRPA analysis, you can use the main ScaledL2NormPool2d
        layer directly. You do not need to call this export function first.
        """
        # This returns a new, standard nn.Module that is also LiRPA-compatible.
        return _ExportedL2Pool(
            kernel_size=self.kernel_size,
            stride=self.stride,
            ceil_mode=self.ceil_mode,
            coeff=self._coefficient_lip
        )

    
class ScaledAdaptiveL2NormPool2d(torch.nn.Module, torchlip.module.LipschitzModule):
    def __init__(
        self,
        output_size: _size_2_t = (1, 1),
        k_coef_lip: float = 1.0,
    ):
        """
        auto_LiRPA-compatible Adaptive L2-norm pooling layer.

        This layer's forward pass is implemented using only operations natively
        supported by auto_LiRPA (pow, adaptive_avg_pool2d, sqrt, mul).
        """
        torch.nn.Module.__init__(self)
        torchlip.module.LipschitzModule.__init__(self, k_coef_lip)
        
        # Ensure output_size is a tuple of two integers
        if not isinstance(output_size, tuple) or len(output_size) != 2:
             output_size = _pair(output_size)

        # For this operation to be a valid norm, it must collapse the spatial dimensions
        if output_size[0] != 1 or output_size[1] != 1:
            raise ValueError("output_size must be (1, 1) for ScaledAdaptiveL2NormPool2d")
        
        self.output_size = output_size
        # We use the standard AdaptiveAvgPool2d as a supported building block
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs adaptive L2 pooling using a sequence of LiRPA-compatible operations.
        """
        # 1. Get spatial dimensions to calculate total number of elements.
        h, w = x.shape[-2:]
        num_elements = h * w

        # 2. Square the input tensor.
        x_squared = torch.pow(x, 2)
        
        # 3. Apply adaptive average pooling to the squared tensor.
        # This computes (sum of squares) / num_elements
        avg_of_squares = self.adaptive_avg_pool(x_squared)
        
        # 4. Multiply by num_elements to get the sum of squares.
        sum_of_squares = avg_of_squares * num_elements
        
        # 5. Take the square root to get the L2 norm over the spatial dimensions.
        # Add a small epsilon for numerical stability.
        pooled = torch.sqrt(sum_of_squares + 1e-9)
        
        # 6. Apply the Lipschitz scaling factor.
        return pooled * self._coefficient_lip
        
    def __repr__(self):
        return (f"ScaledAdaptiveL2NormPool2d(output_size={self.output_size}, "
                f"k_coef_lip={self._coefficient_lip})")

    def vanilla_export(self) -> nn.Module:
        """
        Exports the layer to a self-contained, auto_LiRPA-compatible nn.Module.
        """
        return _ExportedAdaptiveL2Pool(
            output_size=self.output_size,
            coeff=self._coefficient_lip
        )    

class FlattenChannelLast(nn.Module):
    """
    A custom PyTorch module that flattens a tensor by interleaving the channels,
    mimicking the behavior of Keras' `Flatten(data_format="channels_last")` on a
    `channels_first` input tensor.

    It assumes the input tensor is in `channels_first` format (N, C, H, W).
    It works by first permuting the dimensions to `(N, H, W, C)` and then
    flattening the last three dimensions.

    Input Shape: (N, C, H, W)
    Output Shape: (N, C * H * W)
    """
    def __init__(self):
        """
        Initializes the FlattenChannelLast module.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the layer.

        Args:
            x: The input tensor with shape (N, C, H, W).

        Returns:
            The flattened tensor with shape (N, C * H * W).
        """
        # Ensure the input is 4D, which is the expected format
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (got {x.dim()}D input)")

        # Get the batch size to handle variable batch sizes correctly
        batch_size = x.shape[0]

        # The core logic:
        # 1. Permute dimensions from (N, C, H, W) -> (N, H, W, C)
        #    The indices are (0, 2, 3, 1)
        x_permuted = x.permute(0, 2, 3, 1)

        # 2. Flatten the permuted tensor. `reshape` is generally preferred.
        #    The `-1` tells PyTorch to calculate the correct size for that dimension.
        #    This results in a tensor of shape (N, H * W * C).
        return x_permuted.reshape(batch_size, -1)
    
    
def debug_and_compare_submodels(vanilla_model, pytorch_model, test_tensor):
    """
    Compares the output of Keras and PyTorch models layer by layer.

    Args:
        vanilla_model (keras.Model): The full Keras model. UNFOLDED !!
        pytorch_model (torch.nn.Module): The full PyTorch model.
        test_tensor_nchw (torch.Tensor): A test input tensor in NCHW format.
    """
    pt_layers = list(pytorch_model.children())
    kr_layers = vanilla_model.layers
    
    # Ensure the number of layers match.
    # Note: Keras might have an InputLayer at the beginning, which we can skip.
    if isinstance(kr_layers[0], keras.layers.InputLayer):
        kr_layers = kr_layers[1:]
        
    assert len(pt_layers) == len(kr_layers), \
        f"Layer count mismatch! PyTorch: {len(pt_layers)}, Keras: {len(kr_layers)}"

    num_layers = len(pt_layers)
    print(f"\n--- Starting Layer-by-Layer Comparison ({num_layers} layers) ---\n")

    # --- THIS IS THE CORRECTED LINE ---
    # Get the input shape from the model itself, not the first layer.
    input_shape = vanilla_model.input_shape[1:] # e.g., (28, 28, 1)

    for k in range(num_layers, 0, -1):
        print(f"================== COMPARING FIRST {k} LAYERS ==================")

       
        
        # --- Create PyTorch Sub-Model ---
        sub_pt_model = nn.Sequential(*pt_layers[:k])
        sub_pt_model.eval()
        print(sub_pt_model)
        
        # --- Create Keras Sub-Model ---
        # We need to add an Input layer that matches the expected format.
        # The input shape to the first layer in the original model tells us what's needed.
        sub_kr_model = keras.Sequential([keras.layers.Input(shape=input_shape)] + kr_layers[:k])
        sub_kr_model.summary()
        try:
            # --- Get Outputs ---
            pt_output = sub_pt_model(test_tensor)
            kr_output = sub_kr_model(test_tensor)
            
            # Convert Keras output (TensorFlow tensor) to a PyTorch tensor
            kr_output = torch.from_numpy(kr_output.detach().cpu().numpy())

            # Reshape PyTorch output if needed (e.g., after a Conv layer) to match Keras
            # Keras Conv2D output is NHWC, PyTorch is NCHW.
            # We'll flatten both to compare them reliably regardless of shape.
            pt_flat = pt_output.flatten()
            kr_flat = kr_output.flatten()

            # --- Compare and Print ---
            print(f"PyTorch sub-model output shape: {list(pt_output.shape)}")
            print(f"Keras sub-model output shape:   {list(kr_output.shape)}")
            
            # If shapes don't match after a conv layer, it's likely a CWH/HWC issue
            if len(pt_output.shape) == 4 and pt_output.shape[1] != kr_output.shape[1]:
                 # Permute PyTorch from NCHW to NHWC for direct comparison
                 kr_flat = kr_output.permute(0, 3, 1, 2).flatten()


            print(f"\nPyTorch Output (first 8 values): {pt_flat[:8].tolist()}")
            print(f"Keras   Output (first 8 values): {kr_flat[:8].tolist()}")

            # Calculate L2 distance (Euclidean distance)
            distance = torch.dist(pt_flat, kr_flat)
            print(f"\n---> L2 Distance between outputs: {distance.item():.6f}\n")
            # if k==6:
            #     import pdb;pdb.set_trace()

        except Exception as e:
            print(f"!!! Error comparing at k={k}: {e}")
            print("This could be due to an input shape mismatch for this specific sub-model.\n")


def unfold_keras_model(model_to_unfold):
    """
    Rebuilds a Keras model to have separate activation layers.

    Args:
        model_to_unfold (keras.Model): The original Keras model with integrated activations.

    Returns:
        keras.Model: A new model with a 1-to-1 layer structure similar to PyTorch.
    """
    print("--- Unfolding Keras model to separate activation layers ---")
    
    # Use the Functional API to build a new graph
    input_tensor = keras.Input(shape=model_to_unfold.input_shape[1:], name="unfolded_input")
    x = input_tensor
    
    new_layers = []

    for layer in model_to_unfold.layers:
        # Skip the original input layer if it exists
        if isinstance(layer, keras.layers.InputLayer):
            continue
        # --- EXISTING LOGIC FOR ACTIVATIONS ---
        if hasattr(layer, 'activation') and layer.activation is not None and layer.activation != keras.activations.get('linear'):
            print(f"Found and unfolding bundled activation in layer: {layer.name}")
            config = layer.get_config()
            print("1")
            activation_fn_or_layer = layer.activation
            print("2")
            config['activation'] = 'linear'
            print("3")
            base_layer = layer.__class__.from_config(config)
            x = base_layer(x)
            base_layer.set_weights(layer.get_weights())
            x = activation_fn_or_layer(x)
            new_layers.append(base_layer)
            if isinstance(activation_fn_or_layer, keras.layers.Layer):
                 new_layers.append(activation_fn_or_layer)
            else:
                 new_layers.append(keras.layers.Activation(activation_fn_or_layer))
        
        # --- LOGIC FOR ALL OTHER LAYERS ---
        else:
            x = layer(x)
            new_layers.append(layer)

    # Create the new model from the input tensor and the final output tensor 'x'
    unfolded_model = keras.Model(inputs=input_tensor, outputs=x, name="unfolded_model")
    
    print("Unfolded model summary:")
    unfolded_model.summary()

    # The model object is what we need, but we can also return the layer list for the debugger
    return unfolded_model

# https://github.com/keras-team/keras/blob/v3.11.1/keras/src/layers/reshaping/flatten.py#L11  Lines 42-46




def test_flatten():
    x = keras.random.normal((16,7,7))[None]

    print(Flatten(data_format="channels_first")(x)[:,2]) #group 1

    print(Flatten(data_format="channels_last")(x)[:,2]) #group 2

    print(Flatten()(x)[:,2]) #group 1

    print(nn.Flatten()(x)[:,2]) #group 2

    #Channel first
    liste = []
    for i in range(16):
        for j in range(7):
            for k in range(7):
                liste.append(x[:,i, j, k])
    print(liste[2]) #group 2

    print(x.view(-1)[2]) #group2


def evaluate_model(model, device, test_loader):
    """
    Evaluates a trained PyTorch model on a given test dataset.

    Args:
        model (nn.Module): The trained PyTorch model to evaluate.
        device (torch.device): The device to run the evaluation on (e.g., 'cuda' or 'cpu').
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        float: The accuracy of the model on the test set as a percentage.
    """
    # 1. Set the model to evaluation mode
    # This is crucial as it disables layers like Dropout and uses the learned
    # statistics for Batch Normalization.
    model = model.to(device)
    model.eval()

    # 2. Initialize counters
    test_loss = 0
    correct = 0

    # 3. Disable gradient calculations
    # We don't need to calculate gradients for evaluation, which saves memory and computation.
    with torch.no_grad():
        # 4. Iterate over the test data
        for data, target in test_loader:
            # Move data and target tensors to the specified device
            data, target = data.to(device), target.to(device)

            # Perform a forward pass
            output = model(data)

            # Calculate the loss for the batch and add it to the total
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # Get the index of the max log-probability (the predicted class)
            pred = output.argmax(dim=1, keepdim=True)

            # Compare predictions to the true labels and count correct ones
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 5. Calculate average loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    # 6. Print the results
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return accuracy