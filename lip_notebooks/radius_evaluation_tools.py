import numpy as np
import keras.ops as K
import torchattacks
import keras
from decomon.layers import DecomonLayer
from decomon.models import clone
from lipschitz_custom_tools import affine_bound_groupsort_output_keras, affine_bound_sqrt_output_keras, affine_bound_square_output_keras
from decomon.perturbation_domain import BallDomain
from decomon import get_lower_noise, get_range_noise, get_upper_noise
from deel.lip.activations import GroupSort, GroupSort2
from lip_notebooks.LipschitzOptimization.lipschitz_decomon_tools import get_local_maximum, echantillonner_boule_l2_simple

# Compute 1-lip certificates
def compute_certificate(images, model, L=1):    
    values, _ = K.top_k(model(images), k=2)
    certificates = (values[:, 0] - values[:, 1]) / (np.sqrt(2)*L)
    return certificates  
# Compute 1-lip certificate for single output neurons
def compute_binary_certificate(images, model, L=1):    
    values = model(images)[:,0]
    certificates = K.abs(values)/L
    return certificates   
# Compute the starting Point for Dichotomy, it corresponds to the 
# l2 distance with the closest point with different class
def starting_point_dichotomy(idx, images, targets):
    mask_different_classes = targets[idx] != targets
    images_diffferent_classes = images[mask_different_classes]
    score = K.amin((images[idx] - images_diffferent_classes).square().sum(dim=(1, 2, 3)).sqrt())
    return score

def single_compute_optimistic_radius_PGD(idx, images, targets, certificates, model, n_iter = 10):
    image = images[idx:idx+1]
    target = targets[idx:idx+1]
    certificate = certificates[idx:idx+1]
    # We use dichotomy algorithm to fine the smallest optimistic radius
    # We start from the closest point with different class
    eps_working = d_up = starting_point_dichotomy(idx, images, targets)
    d_low = certificate
    # print(d_up, d_low)
    for _ in range(n_iter):
        eps_current = (d_up+d_low)/2
        # atk_van = torchattacks.PGDL2(model, eps=eps_current, alpha=eps_current/5, steps=10, random_start=True)
        atk_van = torchattacks.PGDL2(model, eps=eps_current, alpha=eps_current/5, steps=int((10*eps_current)), random_start=True)
        adv_image = atk_van(image, target)
        # return 0 if the attack doesn't work
        if (K.argmax(model(adv_image), axis=1) == target):
            d_low = eps_current
        else:
            eps_working = d_up = (image - adv_image).square().sum(dim=(1, 2, 3)).sqrt()
    return eps_working

def single_compute_optimistic_radius_AA(idx, images, targets, certificates, model, n_iter = 10):
    image = images[idx:idx+1]
    target = targets[idx:idx+1]
    certificate = certificates[idx:idx+1]
    # We use dichotomy algorithm to fine the smallest optimistic radius
    # We start from the closest point with different class
    eps_working = d_up = starting_point_dichotomy(idx, images, targets)
    d_low = d_low = certificate
    for _ in range(n_iter):
        eps_current = (d_up+d_low)/2
        atk = torchattacks.AutoAttack(model, norm='L2', eps=eps_current)
        adv_image = atk(image, target)
        if (K.argmax(model(adv_image), axis=1) == target):
            d_low = eps_current
        else:
            eps_working = d_up = (image - adv_image).square().sum(dim=(1, 2, 3)).sqrt()
    return eps_working

def single_compute_optimistic_radius_AA_binary(idx, images, targets, certificates, model, n_iter = 10):
    # Give a model with at least 4 outputs ( 2 artificially added)
    image = images[idx:idx+1]
    target = targets[idx:idx+1]
    certificate = certificates[idx:idx+1]
    # We use dichotomy algorithm to fine the smallest optimistic radius
    # We start from the closest point with different class
    eps_working = d_up = starting_point_dichotomy(idx, images, targets)
    d_low = d_low = certificate
    for _ in range(n_iter):
        eps_current = (d_up+d_low)/2
        atk = torchattacks.AutoAttack(model, norm='L2', eps=eps_current, n_classes=2, version="standard")
        adv_image = atk(image, target)
        if (K.argmax(model(adv_image), axis=1) == target):
            d_low = eps_current
        else:
            eps_working = d_up = (image - adv_image).square().sum(dim=(1, 2, 3)).sqrt()
    return eps_working

class DecomonGroupSort2(DecomonLayer):
    layer : GroupSort2
    increasing = True
    def get_affine_bounds(self, lower, upper):
        (W_low, b_low), (W_up, b_up) = affine_bound_groupsort_output_keras(lower, upper)
        W_low = K.transpose(W_low,(0,2,1))
        W_up = K.transpose(W_up,(0,2,1))
        return W_low, b_low, W_up, b_up

def single_compute_decomon_radius(idx, images, targets, model, n_iter = 10):
    image = images[idx:idx+1]
    target = targets[idx:idx+1]
    # certificate = certificates[idx:idx+1]
    # We use dichotomy algorithm to fine the smallest optimistic radius
    # We start from the closest point with different class
    d_up = starting_point_dichotomy(idx, images, targets)
    eps_working = d_low = 0
    for _ in range(n_iter):
        eps_current = (d_up+d_low)/2
        # print(eps_current)
        perturbation_domain = BallDomain(eps=eps_current, p=2)
        decomon_model = clone(model, mapping_keras2decomon_classes={GroupSort2:DecomonGroupSort2}, final_ibp=True, final_affine=False, perturbation_domain=perturbation_domain, method='crown')
        # upper = get_upper_noise(decomon_model,  image.cpu().detach().numpy(), eps=eps_current, p=2)[:, 0]
        # lower = get_lower_noise(decomon_model, image.cpu().detach().numpy(), eps=eps_current, p=2)[:, 0]
        # lower, upper = decomon_model.predict(image, eps=eps_current,p=2)
        lower, upper = decomon_model.predict(image, verbose=0)
        if (target==0 and upper<=0) or (target==1 and lower>=0):
            # print("working", target, upper, lower)
            eps_working = d_low = eps_current
        else:
            # print("not working", target, upper, lower)
            d_up = eps_current
    return eps_working

def single_compute_relaxation_radius(idx, images, targets, model, nb_pts, n_iter = 10):
    image = images[idx:idx+1].flatten().detach().cpu().numpy()
    target = targets[idx:idx+1]

    
    # We use dichotomy algorithm to fine the smallest optimistic radius
    # We start from the closest point with different class
    d_up = starting_point_dichotomy(idx, images, targets).detach().cpu().numpy()
    eps_working = d_low = 0
    for _ in range(n_iter):
        eps_current = (d_up+d_low)/2
        y_list = []
        for i in range(nb_pts):
            y_list.append(echantillonner_boule_l2_simple(image, eps_current))
        _, f_adv = get_local_maximum(image, target, eps_current, y_list, model)

        if (target==0 and f_adv<=0) or (target==1 and f_adv>=0):
            print("working", target, f_adv)
            eps_working = d_low = eps_current
        else:
            print("not working", target, f_adv)
            d_up = eps_current
            
    return eps_working