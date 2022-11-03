import  torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

        
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, kernel_size=(4, 4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(), 
            nn.ConvTranspose2d(64 * 8, 64 * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(), 
            nn.ConvTranspose2d( 64 * 4, 64 * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(), 
            nn.ConvTranspose2d( 64 * 2, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.ConvTranspose2d( 64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Tanh()
        ).apply(weights_init)
        
    def forward(self, x):
        return self.main(x)
        
        
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2), 
            nn.Conv2d(64, 64 * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(negative_slope=0.2), 
            nn.Conv2d(64 * 2, 64 * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(negative_slope=0.2), 
            nn.Conv2d(64 * 4, 64 * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(negative_slope=0.2), 
            nn.Conv2d(64 * 8, 1, kernel_size=(4, 4), stride=(1, 1), bias=False),
            nn.Sigmoid()
        ).apply(weights_init)
        
    def forward(self, x):
        return self.main(x)
