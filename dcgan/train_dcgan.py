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

from dcgan import Generator, Discriminator


def run(params):
    now = datetime.now()
    filename_checkpoint_netG = "../checkpoints/dcgan_netG_{}.ckpt".format(now.strftime("%m%d%H%M%S"))
    filename_checkpoint_netD = "../checkpoints/dcgan_netD_{}.ckpt".format(now.strftime("%m%d%H%M%S"))
    filename_hyperparameters = "../checkpoints/dcgan_{}-params.json".format(now.strftime("%m%d%H%M%S"))

    # Load dataset
    transform = transforms.Compose([
       transforms.Resize(64),
       transforms.CenterCrop(64),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # between [-1, 1]
    ])

    trainset = CelebA('../', split='train', target_type="attr", transform=transform, download=False)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True)

    # Load model
    device = "cuda"
    netG, netD = Generator().to(device), Discriminator().to(device)

    # Training
    criterion = nn.BCELoss()

    optimizerD = optim.Adam(netD.parameters(), lr=params.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=params.lr, betas=(0.5, 0.999))

    for epoch in range(params.num_epochs):
        with tqdm(total=len(trainloader)) as pbar:
            for idx, (images, labels) in enumerate(trainloader):
                images = images.to(device)
                b_size = images.size(0)        

                ## Update discriminator
                netD.zero_grad()

                g_real_predicted_labels = netD(images).view(-1)
                label = torch.ones((b_size,), device=device)
                g_real_error = criterion(g_real_predicted_labels, label)
                g_real_error.backward()
                D_x = g_real_predicted_labels.mean().item()

                noise = torch.randn(b_size, 100, 1, 1, device=device)
                fake = netG(noise)
                g_fake_predicted_labels = netD(fake.detach()).view(-1)
                label.fill_(0.)
                g_fake_error = criterion(g_fake_predicted_labels, label)
                g_fake_error.backward()
                D_G_z1 = g_fake_predicted_labels.mean().item()
                g_error = g_real_error + g_fake_error

                optimizerD.step()

                ## Update generator
                netG.zero_grad()

                d_predicted_labels = netD(fake).view(-1)
                label.fill_(1.0)
                d_error = criterion(d_predicted_labels, label)
                d_error.backward()
                D_G_z2 = d_predicted_labels.mean().item()

                optimizerG.step()

                ## Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "epoch": epoch,
                    "idx": idx,
                    "d_error": round(d_error.item(), 3),
                    "g_error": round(g_error.item(), 3),
                    "D_x": round(D_x, 3),
                    "D(G(z)) pre": round(D_G_z1, 3),
                    "D(G(z)) post": round(D_G_z2, 3),
                })

    # Save
    dictionary = vars(params)
    torch.save(netD.state_dict(), filename_checkpoint_netD)
    torch.save(netG.state_dict(), filename_checkpoint_netG)

    with open(filename_hyperparameters, "w") as f:
        json.dump(dictionary, f)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0002)
    params = parser.parse_args()
    print(params)
    
    run(params)