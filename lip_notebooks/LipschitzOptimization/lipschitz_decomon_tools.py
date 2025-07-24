import numpy as np
from scipy.optimize import minimize
from keras import layers
import keras
import keras.ops as K
import matplotlib.pyplot as plt

def function_to_optimize(x, W, b, y, model, L=1):
    # x (4,)
    output = model(y.reshape((1,28,28))[None]).cpu().detach().numpy()[0,0] +\
          L*np.sqrt(W@x+b) #scalar
    # print(output.shape)
    return output

def function_to_optimize_all(x, label, W_list, b_list, y_list, model, L=1):
    # function we want to optimize, combination of lipschitz constraints in all yi
    outputs = []
    for i in range(len(y_list)):
        if label == 0:
            output = model(y_list[i].reshape((1,28,28))[None]).cpu().detach().numpy()[0,0] +\
                L*np.sqrt(W_list[i]@x+b_list[i]) #scalar
            outputs.append(output)
            # function = np.min(outputs)
            # concave
        else:
            output = model(y_list[i].reshape((1,28,28))[None]).cpu().detach().numpy()[0,0] -\
                L*np.sqrt(W_list[i]@x+b_list[i]) #scalar
            outputs.append(output)
            # function = np.max(outputs)    
            # convexe
    # cp.min ou cp.max doit être appliqué HORS de la boucle
    if label == 0:
        # min(min(concaves)) -> min(concave) -> NON CONVEXE
        function = np.min(outputs) 
    else:
        # min(max(convexes)) -> min(convexe) -> CONVEXE
        function = np.max(outputs)
    return function

def get_argm(x, label, W_list, b_list, y_list, model, L=1):
    # get the argmin or agmax of the function to optimize all
    outputs = []
    for i in range(len(y_list)):
        if label == 0:
            output = model(y_list[i].reshape((1,28,28))[None]).cpu().detach().numpy()[0,0] +\
                L*np.sqrt(W_list[i]@x+b_list[i]) #scalar
            outputs.append(output)
            argm = np.argmin(outputs)
        else:
            output = model(y_list[i].reshape((1,28,28))[None]).cpu().detach().numpy()[0,0] -\
                L*np.sqrt(W_list[i]@x+b_list[i]) #scalar
            outputs.append(output)
            argm = np.argmax(outputs)    
    return argm

def jac_function_to_optimize(x, label, W_list, b_list, y_list, model, L=1):
    # rajouter eps à la racine
    arg = get_argm(x, label, W_list, b_list, y_list, model, L=1)
    if label==0:    
        output = (L*W_list[arg])/(2*np.sqrt(W_list[arg]@x+b_list[arg]))
    else:
        output = -(L*W_list[arg])/(2*np.sqrt(W_list[arg]@x+b_list[arg]))
    return output

def square_backward_bounds(l, u, y):
    # l (4,)
    # u (4,)
    # y (4,)

    u = u - y
    l = l - y

    W = u + l #(4,)
    b = np.sum(-u*l) - W@y #scalar
    return W, np.array(b)[None]#(4,) & (1,)

def echantillonner_boule_l2_simple_surbord(x, epsilon):
    d = x.shape[0] # Dimension

    # 1. Vecteur gaussien aléatoire (direction)
    u = np.random.randn(d)
    norm_u = np.linalg.norm(u)

    # 3. Point final = centre + direction_normalisée * distance
    y = x + epsilon * (u / norm_u)

    return y

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

def get_local_maximum(x, label, eps, y_list, model, L=1):
    # # Define your convex function
    # def f(x):
    #     # Example: quadratic function
    #     return np.dot(x, x) + 3 * x[0] - x[1]  # Replace with your actual function
    x_ball_center = x
    x_ball_center = np.asarray(x_ball_center, dtype=np.float64)
    history_values = []
    # l = x_ball_center-eps
    # u = x_ball_center+eps
    l = x-eps
    u = x+eps

    W_list = []
    b_list = []
    for y_i in y_list:
        W, b = square_backward_bounds(l,u,y_i)
        W_list.append(W)
        b_list.append(b)

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
    
    # def my_callback(intermediate_result):
    #     """
    #     Fonction de callback appelée à chaque itération par minimize.
    #     Elle enregistre la valeur actuelle de la fonction objectif.
    #     """
    #     # print(intermediate_result)
    #     current_fun_value = intermediate_result
    #     history_values.append(current_fun_value)

    args_contrainte = (x_ball_center, eps)
    # Set up the constraint dictionary
    constraints = ({
        'type': 'ineq',  # Inequality constraint: constraint(x) >= 0
        'fun': unit_ball_constraint,
        'jac': jacobian_unit_ball_constraint,
        'args': args_contrainte
    })

    list_test = [echantillonner_boule_l2_simple(x_ball_center, eps) for _ in range(1000)]
    list_exp = [function_to_optimize_all(x_i, label, W_list, b_list, y_list, model, L) for x_i in list_test]
    print("empirical max values", sorted(list_exp)[-3:])
    print("empirical min values", sorted(list_exp)[:3])


    # Run the optimizer
    if label == 0:
        result = minimize(fun=lambda x :-function_to_optimize_all(x, label, W_list, b_list, y_list, model, L),\
        jac= lambda x :-jac_function_to_optimize(x, label, W_list, b_list, y_list, model, L),\
        x0 = x_ball_center, method='SLSQP', constraints=constraints)
    else:
        # options = {
        #     'maxiter': 1000,  # Set your desired maximum number of iterations here
        #     'disp': True      # Set to True to print convergence messages
        # }
        #tester avec maxiter 1, 2, 3, 4
        result = minimize(fun=lambda x :function_to_optimize_all(x, label, W_list, b_list, y_list, model, L),\
        jac= lambda x :jac_function_to_optimize(x, label, W_list, b_list, y_list, model, L),\
        x0 = x_ball_center, method='SLSQP', constraints=constraints, options=options)
        # , callback=my_callback
    # result = minimize(fun=lambda x :-function_to_optimize(x, W_1, b_1, y), x0 = x_ball_center, method='SLSQP', constraints=constraints)
    # attention, le maximum est - result

    # if history_values: # S'assurer que l'historique n'est pas vide
    #     history_fun_values = [function_to_optimize_all(x_i, label, W_list, b_list, y_list, model, L) for x_i in history_values]
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(history_fun_values, marker='o', linestyle='-', color='skyblue')
    #     plt.title('Historique de la valeur de la fonction objectif par itération (SLSQP)')
    #     plt.xlabel('Numéro d\'itération')
    #     plt.ylabel('Valeur de f(x)')
    #     plt.grid(True)
    #     # plt.yscale('log') # Utile si la fonction décroît rapidement
    #     plt.xticks(range(0, len(history_fun_values), max(1, len(history_fun_values)//10))) # Affiche un nombre raisonnable de tics
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print("\nAucun historique de la fonction objectif n'a été enregistré. Le solveur a peut-être convergé en une seule itération ou le callback n'a pas été appelé.")

    # Display results
    if result.success:
        if label == 0:
            return result.x, -result.fun
        #, history_fun_values
        else:
            return result.x, result.fun
        #, history_fun_values
    else:
        print("Optimization failed:", result.message)
        raise ValueError(result.message)
        # return 0, 0, history_fun_values

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

def get_local_maximum_multiclass(x, label, eps, y_list, model, L=1):
    """
    Adaptation du getlocalmaximum au cas multiclasse. On vient borner fgt - fi qui est une fonction racine de 2 lip
    """
    n_classes = model.output_shape[-1]
    list_outputs = list(range(n_classes))
    # print(list_outputs)
    list_outputs.remove(label)
    # print(list_outputs)
    print(K.argsort(model(x.reshape((1,28,28))[None]))[:,-2])
    difference_model = create_difference_model(model, label, K.argsort(model(x.reshape((1,28,28))[None]))[:,-2])

    _, min_one_vs_all, _ =  get_local_maximum(x, 1, eps, y_list, difference_model, L=np.sqrt(2)*L)   
    
    return min_one_vs_all

# def get_local_maximum_multiclass(x, label, eps, y_list, model, L=1):
#     """
#     Adaptation du getlocalmaximum au cas multiclasse. On vient borner fgt - fi qui est une fonction racine de 2 lip
#     """
#     n_classes = model.output_shape[-1]
#     list_outputs = list(range(n_classes))
#     # print(list_outputs)
#     list_outputs.remove(label)
#     # print(list_outputs)
#     current_min = 1000
#     for i in list_outputs:
#         difference_model = create_difference_model(model, label, i)

#         _, max_one_vs_all =  get_local_maximum(x, 1, eps, y_list, difference_model, L=np.sqrt(2)*L)   
#         if max_one_vs_all < current_min:
#             current_min = max_one_vs_all
#     return current_min