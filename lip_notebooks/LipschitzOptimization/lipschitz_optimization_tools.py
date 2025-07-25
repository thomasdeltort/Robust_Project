import numpy as np
from scipy.optimize import minimize
from keras import layers
import keras
import keras.ops as K
import matplotlib.pyplot as plt
import cvxpy as cp


def function_to_optimize_all(x, label, W_list, b_list, y_list, model, L=1):
    # function we want to optimize, combination of lipschitz constraints in all yi
    outputs = []
    for i in range(len(y_list)):
        if label == 0:
            output = model(y_list[i].reshape((1,28,28))[None]).cpu().detach().numpy()[0,0] +\
                L*cp.sqrt(W_list[i]@x+b_list[i]) #scalar
            outputs.append(output)
            # concave
        else:
            output = model(y_list[i].reshape((1,28,28))[None]).cpu().detach().numpy()[0,0] -\
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

def create_difference_model(base_model, label, i):
    """
    Crée et retourne un nouveau modèle Keras qui calcule la différence
    entre le logit du 'label' et le logit de 'i'.
    """
    entree_base = base_model.inputs
    sortie_logits_base = base_model.outputs[0]
    # print(label)
    # Définition de la couche Lambda avec la correction et un nom unique
    difference = layers.Lambda(
        # lambda x, current_i=i: x[:, label] - x[:, current_i],
        lambda z: z[:, label:label+1] - z[:, i:i+1], 
        output_shape=(1,),
        # Nom de couche unique : très important !
        name=f"difference_{label}_vs_{i}"
    )(sortie_logits_base)

    # Création du modèle avec un nom unique
    difference_model = keras.Model(
        inputs=entree_base,
        outputs=difference,
        name=f"model_diff_{label}_vs_{i}"
    )
    
    return difference_model

def get_local_maximum(x_sample, label, eps, y_list, model, L=1):
    l = x_sample-eps
    u = x_sample+eps

    W_list = []
    b_list = []
    for y_i in y_list:
        W, b = square_backward_bounds(l,u,y_i)
        W_list.append(W)
        b_list.append(b)

    x = cp.Variable(784)
    
    constraints = [eps**2 - cp.norm(x - x_sample, 2)**2 >=0]

    # Run the optimizer
    if label == 0:
        obj = cp.Maximize(function_to_optimize_all(x, label, W_list, b_list, y_list, model, L=1))
    else:
        obj = cp.Minimize(function_to_optimize_all(x, label, W_list, b_list, y_list, model, L=1))

    prob = cp.Problem(obj, constraints)
    # prob.solve(solver='CLARABEL', verbose=True)  # Returns the optimal value.
    # prob.solve(solver='ECOS', verbose=True)  # Returns the optimal value.
    prob.solve(solver='SCS', verbose=True)  # Returns the optimal value.
    return prob.status, prob.value, x.value


def get_local_maximum_multiclass(x_sample, label, eps, y_list, model, L=1):
    """
    Adaptation du getlocalmaximum au cas multiclasse. On vient borner fgt - fi qui est une fonction racine de 2 lip
    """
    n_classes = model.output_shape[-1]
    print(type(range(n_classes)))
    list_outputs = list(range(n_classes))
    # print(list_outputs)
    list_outputs.remove(label)
    # print(list_outputs)
    # print(K.argsort(model(x.reshape((1,28,28))[None]))[:,-2])
    difference_model = create_difference_model(model, label, K.argsort(model(x_sample.reshape((1,28,28))[None]))[:,-2])

    return  get_local_maximum(x_sample, 1, eps, y_list, difference_model, L=np.sqrt(2)*L)   