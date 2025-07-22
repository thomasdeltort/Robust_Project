import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_data():
#     # load data
#     batch_size = 128


#     test_transforms = v2.Compose([
#             v2.ToTensor(),
#             # v2.ToDtype(torch.float32, scale=True),
#         ])



#     test_dataset = datasets.CIFAR10(
#             root='./notebooks_creation_models/data', 
#             train=False, 
#             transform=test_transforms
#         )

#     test_loader = DataLoader(
#             dataset=test_dataset,
#             batch_size=batch_size,
#             shuffle=False,
    
#         )
#     return test_loader, test_dataset

def select_data_for_radius_evaluation(test_loader, test_dataset, model, schedulefree=False):
    all_predictions = []
    if not(schedulefree):    
        model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            outputs = model(images.to(device))
            # pdb.set_trace()
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted)
    votre_modele_predictions = torch.cat(all_predictions)

    # print(votre_modele_predictions[0])


    correct_samples_by_class = [[] for _ in range(10)] # 10 listes, une par classe
    # Compteur pour chaque classe afin de respecter la limite de 20 points par classe
    class_counts = torch.zeros(10, dtype=torch.int)

    for i in range(len(test_dataset)):
        image, target = test_dataset[i]
        model_prediction = votre_modele_predictions[i]

        # Si la prédiction est correcte ET que nous n'avons pas encore 20 échantillons pour cette classe
        if model_prediction == target and class_counts[target] < 20:
            correct_samples_by_class[target].append((image, target))
            class_counts[target] += 1

    # Maintenant, nous allons concaténer les échantillons dans l'ordre des classes
    final_images_list = []
    final_targets_list = []

    for class_id in range(10): # Parcourir les classes de 0 à 9
        for image, target in correct_samples_by_class[class_id]:
            final_images_list.append(image)
            final_targets_list.append(target)

    # Conversion des listes en tenseurs PyTorch
    images = torch.stack(final_images_list)
    targets = torch.tensor(final_targets_list)
    return images, targets


def select_data_for_radius_evaluation_saved_points(test_loader, test_dataset, model1, model2, schedulefree=False):
    """
    Selects 200 samples (20 per class) that are correctly classified by two models.
    
    Args:
        test_loader: DataLoader for the test set.
        test_dataset: The test dataset.
        model1: The first trained model.
        model2: The second trained model.
        schedulefree (bool): Flag to skip model.eval().

    Returns:
        A tuple of (images, targets) tensors for the 200 selected samples.
    """
    
    # --- Step 1: Get predictions from both models ---
    all_predictions1 = []
    all_predictions2 = []
    
    # Set both models to evaluation mode
    if not schedulefree:
        model1.eval()
        model2.eval()
    model1 = model1.to(device)
    model2 = model2.to(device)
    
    with torch.no_grad():
        for images, _ in test_loader:
            images_dev = images.to(device)
            
            # Predictions from model1
            outputs1 = model1(images_dev)
            _, predicted1 = torch.max(outputs1.data, 1)
            all_predictions1.append(predicted1)
            
            # Predictions from model2
            outputs2 = model2(images_dev)
            _, predicted2 = torch.max(outputs2.data, 1)
            all_predictions2.append(predicted2)

    # Concatenate predictions for the entire dataset
    model1_predictions = torch.cat(all_predictions1)
    model2_predictions = torch.cat(all_predictions2)

    # --- Step 2: Select samples correctly classified by BOTH models ---
    correct_samples_by_class = [[] for _ in range(10)]  # 10 lists, one for each class
    class_counts = torch.zeros(10, dtype=torch.int)  # Counter to get 20 points per class

    for i in range(len(test_dataset)):
        image, target = test_dataset[i]
        
        # Get predictions for the current sample
        pred1 = model1_predictions[i].item()
        pred2 = model2_predictions[i].item()

        # Check if the prediction from BOTH models is correct AND we need more samples for this class
        if pred1 == target and pred2 == target and class_counts[target] < 20:
            correct_samples_by_class[target].append((image, target))
            class_counts[target] += 1
        
        # Optimization: if we have found 20 samples for every class, we can stop early
        if torch.all(class_counts == 20):
            break

    # --- Step 3: Combine the selected samples into final tensors ---
    final_images_list = []
    final_targets_list = []

    for class_id in range(10):  # Iterate through classes 0 to 9
        for image, target in correct_samples_by_class[class_id]:
            final_images_list.append(image)
            final_targets_list.append(target)

    # Convert lists to PyTorch tensors
    images = torch.stack(final_images_list)
    targets = torch.tensor(final_targets_list, dtype=torch.long)
    
    return images, targets

def load_cifar10(cfg: dict = {}):
    batch_size = cfg['training']['batch_size']
    # batch_size = 128

    # # Initialize transforms
    # train_transforms = v2.Compose([
    #     v2.ToImage(),
    #     # v2.Pad(4),
    #     v2.RandomCrop((32, 32), padding=8),
    #     v2.RandomHorizontalFlip(),
    #     v2.RandomApply([
    #         v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    #     ], p=0.8),
    #     v2.RandAugment(),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize((0.49139968, 0.48215827, 0.44653124),
    #                  (0.24703233, 0.24348505, 0.26158768)),
    #     v2.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')
    # ])
    # test_transforms = v2.Compose([
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize((0.49139968, 0.48215827, 0.44653124), 
    #                  (0.24703233, 0.24348505, 0.26158768))
    # ])
     # Initialize transforms
    train_transforms = v2.Compose([
        v2.ToTensor(),
        # v2.Pad(4),
        v2.RandomCrop((32, 32), padding=8),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        v2.RandAugment(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')
    ])
    test_transforms = v2.Compose([
        v2.ToTensor(),
        # v2.ToDtype(torch.float32, scale=True),
    ])

    # Split dataset into train, calibration, and test sets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        transform=train_transforms,
        download=True
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        transform=test_transforms,
        download=True
    )

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader,  test_loader