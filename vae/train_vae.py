import torch
from torchvision import transforms
from torchvision.datasets import CelebA
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
from datetime import datetime
import json
from tqdm import tqdm

from vae import VariationalAutoEncoder


def run(params):
    now = datetime.now()
    filename_checkpoint = "../checkpoints/vae_{}.ckpt".format(now.strftime("%m%d%H%M%S"))
    filename_hyperparameters = "../checkpoints/vae_{}-params.json".format(now.strftime("%m%d%H%M%S"))

    # Load dataset
    transform = transforms.Compose([
       transforms.Resize(64),
       transforms.CenterCrop(64),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # to have inputs between [-1, 1]
    ])

    trainset = CelebA("../", split='train', target_type="attr", transform=transform, download=False)
    validset = CelebA("../", split='valid', target_type="attr", transform=transform, download=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=params.batch_size, shuffle=False)

    # Load model
    device = "cuda"
    model = VariationalAutoEncoder().to(device)

    # Training
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma_scheduler)

    for epoch in range(params.num_epochs):
        with tqdm(total=len(trainloader)) as pbar:
            for idx, (images, labels) in enumerate(trainloader):

                model.zero_grad()

                images = images.to(device)
                recons, mu, log_var = model(images)
                recons_loss = F.mse_loss(recons, images)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                loss = recons_loss + params.kld_weight * kld_loss            
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix({
                    'epoch': epoch,
                    'idx': idx,
                    'loss': round(loss.detach().cpu().item(), 3),
                    'recons_loss': round(recons_loss.detach().cpu().item(), 3),
                    'kld_loss': round(kld_loss.detach().cpu().item(), 3)               
                })

        with torch.no_grad():
            losses = []
            recons_losses = []
            kld_losses = []
            with tqdm(total=len(validloader)) as pbar:
                for idx, (images, labels) in enumerate(validloader):

                    images = images.to(device)
                    recons, mu, log_var = model(images)
                    recons_loss = F.mse_loss(recons, images).item()
                    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0).item()

                    loss = recons_loss + params.kld_weight * kld_loss   
                    losses.append(loss)
                    recons_losses.append(recons_loss)
                    kld_losses.append(kld_loss)

                    pbar.update(1)
                    pbar.set_postfix({
                        "val_loss": round(sum(losses)/len(losses), 3), 
                        "val_recons_loss": round(sum(recons_losses)/len(recons_losses), 3), 
                        "val_kld_loss": round(sum(kld_losses)/len(kld_losses), 3), 
                    })               

        scheduler.step()

    # Save
    dictionary = vars(params)
    dictionary.update({
        "val_loss": round(sum(losses)/len(losses), 3), 
        "val_recons_loss": round(sum(recons_losses)/len(recons_losses), 3), 
        "val_kld_loss": round(sum(kld_losses)/len(kld_losses), 3), 
    })
    torch.save(model.state_dict(), filename_checkpoint)

    with open(filename_hyperparameters, "w") as f:
        json.dump(dictionary, f)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--gamma_scheduler", type=float, default=0.95)
    parser.add_argument("--kld_weight", type=float, default=0.00025)
    params = parser.parse_args()
    print(params)
    
    run(params)