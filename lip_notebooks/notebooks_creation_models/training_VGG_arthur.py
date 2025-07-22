import time
import argparse
import schedulefree as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
import wandb
import yaml 
from deel import torchlip
from torchinfo import summary
import pdb
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader
from orthogonium.layers.normalization import BatchCentering2D
from VGG_Arthur import load_model, HKRMultiLossLSE


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
        v2.ToDtype(torch.float32, scale=True),
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


def main():
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    args = parser.parse_args()

    # Load configuration file
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load dataset
    train_loader, test_loader = load_cifar10(cfg)

    # Load model
    model = load_model()
    model.to(device)
    summary(model, (1, 3, 32, 32))

    # Retrieve training parameters
    n_epochs = cfg['training']['n_epochs']
    loss_temp = cfg['training']['loss_temp']
    lr = cfg['training']['lr']
    batch_size = cfg['training']['batch_size']


    # Initialize criterion and optimizer
    # criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion = HKRMultiLossLSE(alpha=750, temperature=loss_temp, penalty=0.5, margin=1.0)
    num_steps_per_epoch = len(train_loader.dataset) / batch_size
    warmup_steps = 6 * num_steps_per_epoch
    optimizer = sf.AdamWScheduleFree(model.parameters(), lr=lr, warmup_steps=warmup_steps)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #with wandb.init(project='Lipschitz-Benchmark', config=cfg):
    print(f"Epoch\tTrain Accuracy\tTrain Loss\tTest Accuracy\tTest Loss\tTime")
    for epoch in range(n_epochs):
        start_time = time.time()

        ############################ Train loop ############################
        model.train()
        optimizer.train()
        total_correct, total_loss = 0, 0
        #all_Logits, all_labels = [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels_onehot = nn.functional.one_hot(labels, 10)
            # Forward pass
            logits = model(images)
            #all_Logits.append(logits)
            #all_labels.append(labels)
            # Backward pass
            # pdb.set_trace()
            loss = criterion(logits, labels_onehot)
            loss.backward()
            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_loss += loss.item()


        train_accuracy = 100 * total_correct / len(train_loader.dataset)
        train_loss = total_loss / len(train_loader.dataset)
           
        ############################ Batch_norm calibration #############################    
        # https://github.com/facebookresearch/schedule_free
        # Switch to the averaged weights collected by the optimizer
        optimizer.eval()

        # Switch model to train() mode to UPDATE the BN stats
        # but use no_grad() to NOT update the weights
        model.train() 

        with torch.no_grad():
            # Iterate over the training data to re-calculate running_mean and running_var
            # for the averaged weights. One epoch is usually sufficient.
            for inputs, _ in train_loader:
                model(inputs.to(device)) # Forward pass is enough to update BN stats
            
        print("BatchNorm statistics updated.")

        ############################ Test loop #############################
        model.eval()
        total_correct,total_loss = 0, 0
        #all_logits, all_labels = [], []
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels_onehot = nn.functional.one_hot(labels, 10)

            with torch.no_grad():
                logits = model(images)

                loss = criterion(logits, labels_onehot)

                #all_logits.append(logits)
                #all_labels.append(labels)

                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                #total_adv_correct += (adv_logits.argmax(dim=1) == labels).sum().item()
                total_loss += loss.item()
            #all_logits = torch.cat(all_logits)
            #all_labels = torch.cat(all_labels)

        test_accuracy = 100 * total_correct / len(test_loader.dataset)
        #test_aa_accuracy = 100 * total_adv_correct / len(test_loader.dataset)
        test_loss = total_loss / len(test_loader.dataset)
           
            
        end_time = time.time()
        time_elapsed = end_time - start_time
            
     

        # Print metrics
        print(f"{epoch+1}/{n_epochs}\t"
                f"{train_accuracy:.2f}%\t\t"
                f"{train_loss:.4f}\t\t"
                f"{test_accuracy:.2f}%\t\t"
                f"{test_loss:.4f}\t\t"
                f"{time_elapsed:.2f}s")

    torch.save(model.state_dict(), 'Vgg_lip_1.pt')    
    torch.save(model.vanilla_export().state_dict(), 'Vgg_lip_1_van.pt')  

    return 


if __name__ == "__main__":
    main()