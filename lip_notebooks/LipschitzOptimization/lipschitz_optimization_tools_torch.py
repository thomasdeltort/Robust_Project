import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cvxpy as cp
# torch.from_numpy(x_sample.reshape(input_shape)[None]).to(device)

def function_to_optimize_all(x, label, W_list, b_list, y_list, model, device, input_shape = (1,28,28), L=1):
    # function we want to optimize, combination of lipschitz constraints in all yi
    outputs = []
    for i in range(len(y_list)):
        if label == 0:
            output = model(torch.from_numpy(y_list[i].reshape(input_shape)[None]).float().to(device)).cpu().detach().numpy()[0,0] +\
                L*cp.sqrt(W_list[i]@x+b_list[i]) #scalar
            outputs.append(output)
            # concave
        else:
            output = model(torch.from_numpy(y_list[i].reshape(input_shape)[None]).float().to(device)).cpu().detach().numpy()[0,0] -\
                L*cp.sqrt(W_list[i]@x+b_list[i]) #scalar
            outputs.append(output)   
            # convexe
    if label == 0:
        # min(min(concaves)) -> min(concave) -> NON CONVEXE
        function = cp.min(cp.hstack(outputs)) 
    else:
        # min(max(convexes)) -> min(convexe) -> CONVEXE
        function = cp.max(cp.hstack(outputs))
    return function


def square_backward_bounds(l, u, y):
    # l (4,)
    # u (4,)
    # y (4,)

    u = u - y
    l = l - y

    W = u + l #(4,)
    b = np.sum(-u*l) - W@y #scalar
    return W, np.array(b)[None]#(4,) & (1,)

def echantillonner_boule_l2_simple(x, epsilon, uniform = False):
    d = x.shape[0] # Dimension

    # 1. Vecteur gaussien aléatoire (direction)
    u = np.random.randn(d)
    norm_u = np.linalg.norm(u)

    
    # 2. Distance radiale (avec échelle pour uniformité en volume)
    s = np.random.rand() # Échantillon uniforme dans [0, 1)
    if uniform:
        r = epsilon * s**(1/d) 
    else:
        r = epsilon * s
    # 3. Point final = centre + direction_normalisée * distance
    y = x + r * (u / norm_u)

    return y

import torch
import torch.nn as nn

class DifferenceModel(nn.Module):
    """
    Creates a new PyTorch model that calculates the difference
    between the logit of 'label' and the logit of 'i'.

    This is the PyTorch equivalent of the provided Keras function.
    """
    def __init__(self, base_model, label, i):
        """
        Args:
            base_model (nn.Module): The base model that outputs logits.
            label (int): The index of the first logit.
            i (int): The index of the second logit.
        """
        super().__init__()
        self.base_model = base_model
        self.label = label
        self.i = i

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor for the base_model.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) containing the
                          difference between the specified logits.
        """
        # Get the logits from the base model
        logits = self.base_model(x)

        # Calculate the difference. The slicing [:, index:index+1] is a
        # direct translation of the Keras lambda function's slicing,
        # ensuring the output shape is (batch_size, 1).
        difference = logits[:, self.label:self.label+1] - logits[:, self.i:self.i+1]

        return difference

def get_local_maximum(x_sample, label, eps, y_list, model, device, input_shape = (1,28,28), L=1):
    l = x_sample-eps
    u = x_sample+eps

    W_list = []
    b_list = []
    for y_i in y_list:
        W, b = square_backward_bounds(l,u,y_i)
        W_list.append(W)
        b_list.append(b)

    x = cp.Variable(np.prod(input_shape))
    
    constraints = [eps**2 - cp.norm(x - x_sample, 2)**2 >=0]

    # Run the optimizer
    if label == 0:
        obj = cp.Maximize(function_to_optimize_all(x, label, W_list, b_list, y_list, model, device, input_shape, L=L))
    else:
        obj = cp.Minimize(function_to_optimize_all(x, label, W_list, b_list, y_list, model, device, input_shape, L=L))

    prob = cp.Problem(obj, constraints)
    # prob.solve(solver='CLARABEL', verbose=True)  # Returns the optimal value.
    # prob.solve(solver='ECOS', verbose=False)  # Returns the optimal value.
    prob.solve(solver='SCS', verbose=False)  # Returns the optimal value.
    return prob.status, prob.value, x.value


def get_local_maximum_multiclass(x_sample, label, eps, y_list, model, device, input_shape = (1,28,28), L=1):
    """
    Adaptation du getlocalmaximum au cas multiclasse. On vient borner fgt - fi qui est une fonction racine de 2 lip
    """
    n_classes = model(torch.from_numpy(x_sample.reshape(input_shape)[None]).to(device)).shape[-1]
    list_outputs = list(range(n_classes))
    # print(list_outputs)
    list_outputs.remove(label)
    # print(list_outputs)
    # print(K.argsort(model(x.reshape((1,28,28))[None]))[:,-2])
    difference_model = DifferenceModel(model, label, torch.argsort(model(torch.from_numpy(x_sample.reshape(input_shape)[None]).to(device)))[:,-2])

    return  get_local_maximum(x_sample, 1, eps, y_list, difference_model, device = device, input_shape = input_shape, L=np.sqrt(2)*L)  

# def get_local_maximum_multiclass_CIFAR10(x_sample, label, eps, y_list, model, L=1):
#     """
#     Adaptation du getlocalmaximum au cas multiclasse. On vient borner fgt - fi qui est une fonction racine de 2 lip
#     """
#     n_classes = model.output_shape[-1]
#     list_outputs = list(range(n_classes))
#     # print(list_outputs)
#     list_outputs.remove(label)
#     # print(list_outputs)
#     # print(K.argsort(model(x.reshape((1,28,28))[None]))[:,-2])
#     difference_model = create_difference_model(model, label, K.argsort(model(x_sample.reshape((3,32,32))[None]))[:,-2])

#     return  get_local_maximum(x_sample, 1, eps, y_list, difference_model, L=np.sqrt(2)*L)    