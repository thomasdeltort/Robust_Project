import keras
import keras.ops as K
import numpy as np
from lipschitz_decomon_tools import  echantillonner_boule_l2_simple, echantillonner_boule_l2_simple_surbord
from scipy.optimize import minimize
from keras import layers

# BATCH_SIZE = 128
EPOCHS = 300  # Nombre d'époques (ajustable)
STEPS_PER_EPOCH = 50 # Nombre de batches par époque (car le générateur est infini)
LEARNING_RATE = 0.01
LAMBDA_PENALTY_TF = 100 # Coefficient de pénalité

def norm(x,y):
    return np.linalg.norm(x-y)

def create_dataset(nb, x, yi, eps):    
    input = []
    label = []
    for _ in range(nb):
        x_current = echantillonner_boule_l2_simple(x,eps)
        input.append(x_current)
        label.append(norm(x_current, yi))
    return np.array(input), np.array(label)

def create_dataset_bord(nb, x, yi, eps):    
    input = []
    label = []
    for _ in range(nb):
        x_current = echantillonner_boule_l2_simple_surbord(x,eps)
        input.append(x_current)
        label.append(norm(x_current, yi))
    return np.array(input), np.array(label)

# --- 1. Définition du Modèle Affine (Keras) ---
def create_affine_model(input_dim_model):
    model = keras.Sequential([keras.layers.Input(input_dim_model),
        keras.layers.Dense(1, activation=None, name="affine_layer")
    ], name="simple_affine_network")
    return model

def sur_approximation_mse_loss(y_true_norm, y_pred_affine):
    # y_pred_affine: Sortie du modèle (Wx + b_nn), shape [batch_size, 1]
    # y_true_norm: Norme L2 cible (||x - x0||), shape [batch_size,]
    
    y_pred_affine_squeezed = K.squeeze(y_pred_affine) # Shape [batch_size,]
    
    # Terme 1 (MSE): (g_i - f_i)^2, où g_i = y_pred_affine, f_i = y_true_norm
    mse_gap_term = K.square(y_pred_affine_squeezed - y_true_norm)
    
    # Terme 2 (Pénalité): lambda * ReLU(f_i - g_i)^2
    violation = y_true_norm - y_pred_affine_squeezed # Positif si g_i < f_i (violation)
    penalty_term = LAMBDA_PENALTY_TF * K.square(K.relu(violation))
    
    # Perte moyenne sur le batch
    loss = K.mean(mse_gap_term + penalty_term)
    return loss

def function_to_optimize(x, x0, yi, W, b, eps):
    z0 = x0-yi
    return np.linalg.norm(z0) + eps + 2*z0@x - W@(z0 + x + yi) - b

def generating_hyperplans(x_0, y_list, eps, optimization=False, n_data=2048, n_epochs=10, lr=10e-3, steps_per_epochs=50):
    W_list = []
    b_list = []
    for y_i in y_list:
        inputs, labels = create_dataset_bord(n_data, x_0, y_i, eps)
        model = create_affine_model(x_0.shape)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss=sur_approximation_mse_loss)
        model.fit(inputs, labels,
                    epochs=n_epochs,
                    steps_per_epoch=steps_per_epochs,
                    verbose=0) # verbose=1 pour la barre de progression
        W, b = model.get_weights()
        
        if optimization:
            _, gap = optimize_hyperplane(x_0, y_i, W, b, eps)
            b = b + gap
            
        W_list.append(W.squeeze(1))
        b_list.append(b)
    return W_list, b_list

def optimize_hyperplane(x0, yi, W, b, eps):
    # Define the constraint: ||x - x_centre||_2**2 <= eps**2
    def unit_ball_constraint(x, x_ball_center, eps):
        return eps - np.linalg.norm(x - x_ball_center)

    
    x_ball_center = x0
    x_ball_center = np.asarray(x_ball_center, dtype=np.float64)

    args_contrainte = (x_ball_center, eps)
    # Set up the constraint dictionary
    constraints = ({
        'type': 'eq', 
        'fun': unit_ball_constraint,
        # 'jac': jacobian_unit_ball_constraint,
        'args': args_contrainte
    })
    result = minimize(fun=lambda x :-function_to_optimize(x, x_ball_center, yi, W.squeeze(1), b, eps),\
        # jac= lambda x :-jac_function_to_optimize(x, label, W_list, b_list, y_list, model, L),\
        x0 = x_ball_center, method='SLSQP', constraints=constraints)
    if result.success:
        return result.x, -result.fun
    else:
        print("Optimization failed:", result.message)
        return 0   
# import pdb
def f(z, W, b, y, label, model, L=1):
# Starting from W,b computed by training, we generate the lip bounding function.
    if label==0:
        # print(model(y.reshape((1,28,28))[None]).cpu().detach().numpy())
        # pdb.set_trace()

        return model(y.reshape((1,28,28))[None]).cpu().detach().numpy()[0,0] +\
                L*(W@z +b) #scalar
    else:
        return model(y.reshape((1,28,28))[None]).cpu().detach().numpy()[0,0] -\
                L*(W@z +b) #scalar

def f_all(z, W_list, b_list, y_list, label, model, L=1):
# Compute the supremum of f (depending on the label) over all yi
    output = []
    for i in range(len(y_list)):
        output.append(f(z,W_list[i], b_list[i], y_list[i], label, model, L))
    if label==0:
        return np.max(output)
    else:
        return np.min(output)

def function_to_optimize_all(z, label, x_0, y_list, eps, optimization, model, n_data, n_epochs, L):
    W_list, b_list = generating_hyperplans(x_0, y_list, eps, optimization, n_data, n_epochs)
    return f_all(z, W_list, b_list, y_list, label, model, L)

def get_local_maximum_Learned_Hyperplane(x, label, eps, y_list, model, optimization=False, L=1, n_data=2048, n_epochs=10, lr=10e-3, steps_per_epochs=50):

    # # Define your convex function
    # def f(x):
    #     # Example: quadratic function
    #     return np.dot(x, x) + 3 * x[0] - x[1]  # Replace with your actual function
    x_ball_center = x
    x_ball_center = np.asarray(x_ball_center, dtype=np.float64)

    W_list, b_list = generating_hyperplans(x, y_list, eps, optimization, n_data, n_epochs, lr, steps_per_epochs)

    # Define the constraint: ||x - x_centre||_2**2 <= eps**2
    def unit_ball_constraint(x, x_ball_center, eps):
        return eps**2 - np.linalg.norm(x - x_ball_center)**2

    def jacobian_unit_ball_constraint(x, x_ball_center, eps):
        """
        Jacobien (gradient) de la fonction unit_ball_constraint.
        Retourne -x / ||x||_2.
        Non défini à x = 0.
        """
        # norm_x = np.linalg.norm(x)
        # return -x / norm_x
        return -2*(x - x_ball_center)

    args_contrainte = (x_ball_center, eps)
    # Set up the constraint dictionary
    constraints = ({
        'type': 'ineq',  # Inequality constraint: constraint(x) >= 0
        'fun': unit_ball_constraint,
        'jac': jacobian_unit_ball_constraint,
        'args': args_contrainte
    })

    # Run the optimizer
    if label == 0:
        result = minimize(fun=lambda x :-f_all(x, W_list, b_list, y_list, label, model, L),\
        # jac= lambda x :-jac_function_to_optimize(x, label, W_list, b_list, y_list, model, L),\
        x0 = x_ball_center, method='SLSQP', constraints=constraints)
    else:
        result = minimize(fun=lambda x :f_all(x, W_list, b_list, y_list, label, model, L),\
        # jac= lambda x :jac_function_to_optimize(x, label, W_list, b_list, y_list, model, L),\
        x0 = x_ball_center, method='SLSQP', constraints=constraints)
    # result = minimize(fun=lambda x :-function_to_optimize(x, W_1, b_1, y), x0 = x_ball_center, method='SLSQP', constraints=constraints)
    # attention, le maximum est - result
    # Display results
    if result.success:
        if label == 0:
            return result.x, -result.fun
        else:
            return result.x, result.fun
    else:
        print("Optimization failed:", result.message)
        raise ValueError(result.message)