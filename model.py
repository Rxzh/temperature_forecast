import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, image_channels=1):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 1, kernel_size=4, stride=2, padding=1)  # 192x160
        self.conv2 = nn.Conv2d(1, 2, kernel_size=4, stride=2, padding=1)  # 96x80
        self.conv3 = nn.Conv2d(2, 4, kernel_size=4, stride=2, padding=1)  # 48x40
        self.fc = nn.Linear(4 * 48 * 40, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=4):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=4, stride=2, padding=1)  # 192x160
        self.conv2 = nn.Conv2d(1, 2, kernel_size=4, stride=2, padding=1)  # 96x80
        self.conv3 = nn.Conv2d(2, 4, kernel_size=4, stride=2, padding=1)  # 48x40
        self.fc_mu = nn.Linear(4 * 48 * 40, latent_dim)
        self.fc_logvar = nn.Linear(4 * 48 * 40, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=4, out_channels=1):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 4 * 48 * 40)
        self.deconv1 = nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1)  # 96x80
        self.deconv2 = nn.ConvTranspose2d(2, 1, kernel_size=4, stride=2, padding=1)  # 192x160
        self.deconv3 = nn.ConvTranspose2d(1, out_channels, kernel_size=4, stride=2, padding=1)  # 384x320
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 4, 48, 40)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Sigmoid to get output in range [0, 1]
        return x

class VAE(nn.Module):
    def __init__(self, image_channels=1, latent_dim=4, device='cpu'):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels=image_channels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=image_channels)
        self.discriminator = Discriminator(image_channels=image_channels)
        self.device = device
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar
    
    def _run_epoch(self, batch, optimizer, criterion, lambda_rec=1, lambda_adv=1, lambda_kl=1):
            self.train()
            optimizer.zero_grad()
            x = batch.to(self.device)
            x_reconstructed, mu, logvar = self.forward(x)

            # Reconstruction loss
            rec_loss = criterion(x_reconstructed, x)

            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Adversarial loss
            real_labels = torch.ones(batch.size(0), 1).to(self.device)
            fake_labels = torch.zeros(batch.size(0), 1).to(self.device)

            real_outputs = self.discriminator(x)
            fake_outputs = self.discriminator(x_reconstructed.detach())

            d_loss_real = F.binary_cross_entropy(real_outputs, real_labels)
            d_loss_fake = F.binary_cross_entropy(fake_outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            g_loss = F.binary_cross_entropy(fake_outputs, real_labels)

            # Total VAE loss
            vae_loss = lambda_rec * rec_loss + lambda_adv * g_loss + lambda_kl * kl_loss
            vae_loss.backward()
            optimizer.step()

            return vae_loss.item(), rec_loss.item(), kl_loss.item(), g_loss.item()


