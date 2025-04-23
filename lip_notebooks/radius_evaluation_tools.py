import numpy as np
import keras.ops as K
import torchattacks
import keras

# Compute 1-lip certificates
def compute_certificate(images, model, L=1):    
    values, _ = K.top_k(model(images), k=2)
    certificates = (values[:, 0] - values[:, 1]) / (np.sqrt(2)*L)
    return certificates  
# Compute the starting Point for Dichotomy, it corresponds to the 
# l2 distance with the closest point with different class
def starting_point_dichotomy(idx, images, targets):
    mask_different_classes = targets[idx] != targets
    images_diffferent_classes = images[mask_different_classes]
    return K.amin((images[idx] - images_diffferent_classes).square().sum(dim=(1, 2, 3)).sqrt())

def single_compute_optimistic_radius_PGD(idx, images, targets, certificates, model, n_iter = 10):
    image = images[idx:idx+1]
    target = targets[idx:idx+1]
    certificate = certificates[idx:idx+1]
    # We use dichotomy algorithm to fine the smallest optimistic radius
    # We start from the closest point with different class
    eps_working = d_up = starting_point_dichotomy(1, images, targets)
    d_low = certificate
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
    eps_working = d_up = starting_point_dichotomy(1, images, targets)
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
    eps_working = d_up = starting_point_dichotomy(1, images, targets)
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
