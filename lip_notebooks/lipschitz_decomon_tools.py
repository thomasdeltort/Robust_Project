import numpy as np
from scipy.optimize import minimize

def function_to_optimize(x, W, b, y, model, L=1):
    # x (4,)
    output = model(y.reshape((1,28,28))[None]).cpu().detach().numpy()[0,0] +\
          L*np.sqrt(W@x+b) #scalar
    # print(output.shape)
    return output

def function_to_optimize_all(x, W_list, b_list, y_list, model, L=1):
    # x (4,)
    outputs = []
    for i in range(len(y_list)):
        output = model(y_list[i].reshape((1,28,28))[None]).cpu().detach().numpy()[0,0] +\
          L*np.sqrt(W_list[i]@x+b_list[i]) #scalar
        outputs.append(output)
        
    # print(output.shape)
    return np.min(outputs)

def jac_function_to_optimize(x, W, b, L=1):
    output = L/(2*np.sqrt(W@x+b))*W
    # print(output.shape)
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



def echantillonner_boule_l2_simple(x, epsilon):
  d = x.shape[0] # Dimension

  # 1. Vecteur gaussien aléatoire (direction)
  u = np.random.randn(d)
  norm_u = np.linalg.norm(u)

  
  # 2. Distance radiale (avec échelle pour uniformité en volume)
  s = np.random.rand() # Échantillon uniforme dans [0, 1)
  r = epsilon * s 

  # 3. Point final = centre + direction_normalisée * distance
  y = x + r * (u / norm_u)

  return y

def get_local_maximum(x, eps, y_list, model, L=1):
    # # Define your convex function
    # def f(x):
    #     # Example: quadratic function
    #     return np.dot(x, x) + 3 * x[0] - x[1]  # Replace with your actual function
    x_ball_center = x
    x_ball_center = np.asarray(x_ball_center, dtype=np.float64)

    l = x-eps
    u = x+eps

    W_list = []
    b_list = []
    for y_i in y_list:
        W_1, b_1 = square_backward_bounds(l,u,y_i)
        W_list.append(W_1)
        b_list.append(b_1)

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
    result = minimize(fun=lambda x :-function_to_optimize(x, W_1, b_1, y, model, L),\
                    jac= lambda x :-jac_function_to_optimize(x, W_1, b_1, L),\
                    x0 = x_ball_center, method='SLSQP', constraints=constraints)
    # result = minimize(fun=lambda x :-function_to_optimize(x, W_1, b_1, y), x0 = x_ball_center, method='SLSQP', constraints=constraints)
    # attention, le maximum est - result
    # Display results
    if result.success:
        return result.x, result.fun
    else:
        print("Optimization failed:", result.message)
        raise ValueError(result.message)
