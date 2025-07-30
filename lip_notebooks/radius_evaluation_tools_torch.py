import torch
import numpy as np
import torchattacks
# import pdb
from LipschitzOptimization.lipschitz_optimization_tools_torch import get_local_maximum
from LipschitzOptimization.lipschitz_optimization_tools_torch import get_local_maximum_multiclass, echantillonner_boule_l2_simple, DifferenceModel

# calcul certificat
def compute_certificate(images, model, L=1):    
    values, _ = torch.topk(model(images), k=2)
    certificates = (values[:, 0] - values[:, 1]) / (np.sqrt(2)*L)
    return certificates

# Compute 1-lip certificate for single output neurons
def compute_binary_certificate(images, model, L=1):    
    values = model(images)[:,0]
    certificates = torch.abs(values)/L
    return certificates   
 
# Compute the lip certificates of LiResNet
def compute_certificate_LiResNet(images, model):   
    lc = model.sub_lipschitz().item()
    head = model.head.get_weight()
    pred = torch.argmax(model(images), dim=1)

    batch_radius = []
    for b,j in enumerate(pred):
        
        head_j = head[j:j+1].unsqueeze(1) # batch, 1, dim
        head_ji = head_j - head.unsqueeze(0)  # batch, num_class, dim
        head_ji = head_ji.norm(dim=-1)  # batch, num_class
        lc_global = lc * head_ji
        #non // part
        current_radius = (model(images)[b,j] - model(images)[b,:])/lc_global
        current_radius = torch.cat((current_radius[:,:j], current_radius[:,j+1:]), dim = 1).min().cpu().detach().numpy()
        batch_radius.append(current_radius.item())
    return torch.tensor(batch_radius).unsqueeze(1)



# Dichotomie
def starting_point_dichotomy(idx, images, targets, return_attack=False):
    mask_different_classes = targets[idx] != targets
    images_diffferent_classes = images[mask_different_classes]
    score = torch.min((images[idx] - images_diffferent_classes).square().sum(dim=(1, 2, 3)).sqrt())
    if return_attack:
        idx_attack = torch.argmin((images[idx] - images_diffferent_classes).square().sum(dim=(1, 2, 3)).sqrt())
        attack = images[idx_attack : idx_attack+1]
        return score, attack
    return score

def single_compute_optimistic_radius_PGD(idx, images, targets, certificates, model, return_attack = False, n_iter = 10):
    image = images[idx:idx+1]
    target = targets[idx:idx+1]
    certificate = certificates[idx:idx+1]
    print(certificate)
    # We use dichotomy algorithm to fine the smallest optimistic radius
    # We start from the closest point with different class
    # eps_working = d_up = starting_point_dichotomy(idx, images, targets)
    # d_low = certificate
    if return_attack:
        eps_working, adv_image_working = starting_point_dichotomy(idx, images, targets, return_attack=return_attack)
        d_up = eps_working
    else:
        eps_working = d_up = starting_point_dichotomy(idx, images, targets, return_attack=return_attack)
    d_low = 0
    # pdb.set_trace()
    print('eps_working', eps_working, "d_low", d_low)
    # print(d_up, d_low)
    for _ in range(n_iter):
        eps_current = (d_up+d_low)/2
        # print(d_low.item(), d_up.item(), eps_current.item(), eps_working.item())
      
        # atk_van = torchattacks.PGDL2(model, eps=eps_current, alpha=eps_current/5, steps=10, random_start=True)
        # print("eps_current", eps_current)
        atk_van = torchattacks.PGDL2(model, eps=eps_current, alpha=eps_current/5, steps=int((10*eps_current)), random_start=True)
        adv_image = atk_van(image, target)
        # return 0 if the attack doesn't work
        # print(torch.sqrt(torch.sum(torch.square(image - adv_image), dim=(1, 2, 3))), torch.linalg.norm(image - adv_image))
        if (torch.argmax(model(adv_image)) == target):
            # print("notwarking")
            d_low = eps_current
        else:
            eps_working = d_up = torch.linalg.norm(image - adv_image)
            adv_image_working = adv_image
            # print("working with ", eps_working)
    if return_attack:
        return eps_working, adv_image_working
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
        if (torch.argmax(model(adv_image)) == target):
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
        if (torch.argmax(model(adv_image), axis=1) == target):
            d_low = eps_current
        else:
            eps_working = d_up = (image - adv_image).square().sum(dim=(1, 2, 3)).sqrt()
    return eps_working


def single_compute_relaxation_radius(idx, images, targets, model, nb_pts, device, n_iter = 10, input_shape = (1,28,28)):
    image = images[idx:idx+1].flatten().detach().cpu().numpy()
    target = targets[idx:idx+1]

    
    # We use dichotomy algorithm to fine the smallest optimistic radius
    # We start from the closest point with different class
    d_up = starting_point_dichotomy(idx, images, targets).detach().cpu().numpy()
    eps_working = d_low = 0
    for _ in range(n_iter):
        eps_current = (d_up+d_low)/2
        y_list = []
        for _ in range(nb_pts):
            y_list.append(echantillonner_boule_l2_simple(image, eps_current))
        _, optimum, _ = get_local_maximum(image, target, eps_current, y_list, model, device=device, input_shape = input_shape)

        # print("Status : ", status)

        if (target==0 and optimum<=0) or (target==1 and optimum>=0):
            # print("working", eps_current, optimum)
            eps_working = d_low = eps_current
        else:
            # print("not working", eps_current, optimum)
            d_up = eps_current
        # print("point ", idx, "eps working : ", eps_working)
            
    return eps_working


def single_compute_relaxation_radius_accuracy(idx, images, targets, model, nb_pts, device, input_shape = (1,28,28), lip_certificate=0):
    image = images[idx:idx+1].flatten().detach().cpu().numpy()
    target = targets[idx:idx+1]
    lip_certificate = lip_certificate[idx:idx+1]

    eps_current = lip_certificate.detach().cpu().numpy()
    y_list = []
    for _ in range(nb_pts):
        y_list.append(echantillonner_boule_l2_simple(image, eps_current))
    status, optimum, _ = get_local_maximum(image, target, eps_current, y_list, model, device=device, input_shape = input_shape)

    print('Status : ', status, optimum)

    if (target==0 and optimum<=0) or (target==1 and optimum>=0):
        # print("working", eps_current, optimum)
        result = 1
    else:
        # print("not working", eps_current, optimum)
        result = 0
    # print("point ", idx, "eps working : ", eps_working)
            
    return result


def single_compute_relaxation_radius_multiclass(idx, images, targets, model, nb_pts, device, n_iter = 10, input_shape = (1,28,28)):
    image = images[idx:idx+1].flatten().detach().cpu().numpy()
    target = targets[idx:idx+1]

    
    # We use dichotomy algorithm to fine the smallest optimistic radius
    # We start from the closest point with different class
    d_up = starting_point_dichotomy(idx, images, targets).detach().cpu().numpy()
    eps_working = d_low = 0
    for _ in range(n_iter):
        eps_current = (d_up+d_low)/2
        y_list = []
        for _ in range(nb_pts):
            y_list.append(echantillonner_boule_l2_simple(image, eps_current))
        status, optimum, _ = get_local_maximum_multiclass(image, target, eps_current, y_list, model, device=device, input_shape = input_shape)
        print('Status : ', status, optimum)
        if  (optimum>=0):
            print("working", eps_current, optimum)
            eps_working = d_low = eps_current
        else:
            print("not working", eps_current, optimum)
            d_up = eps_current
        print("point ", idx, "eps working : ", eps_working)
            
    return eps_working

def single_compute_relaxation_radius_multiclass_accuracy(idx, images, targets, model, nb_pts, device, input_shape = (1,28,28), lip_certificate=0):
    image = images[idx:idx+1].flatten().detach().cpu().numpy()
    target = targets[idx:idx+1]
    lip_certificate = lip_certificate[idx:idx+1]

    eps_current = lip_certificate.detach().cpu().numpy()
    print("certificate ", lip_certificate)
    y_list = []
    for _ in range(nb_pts):
        y_list.append(echantillonner_boule_l2_simple(image, eps_current))
    status, optimum, _ = get_local_maximum_multiclass(image, target, eps_current, y_list, model, device=device, input_shape = input_shape)
    print('Status : ', status, optimum)
    if  (optimum>=0):
        result=1
    else:
            # print("not working", eps_current, optimum)
       result=0  
    return result