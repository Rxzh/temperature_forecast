import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import torch
import torch.nn.functional as F
import csv

class Trainer:
    def __init__(self, vae, 
                 discriminator, 
                 train_loader, 
                 val_loader, 
                 criterion, 
                 vae_optimizer, 
                 discriminator_optimizer, args=None):
        
        self.args = args
        self.vae = vae
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.vae_optimizer = vae_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.device = self.args.device
        self.lambda_rec = self.args.lambda_rec
        self.lambda_adv = self.args.lambda_adv
        self.lambda_kl = self.args.lambda_kl

        if not os.path.exists(self.args.ckpt_folder):
            os.makedirs(self.args.ckpt_folder)

        with open(self.args.logfile, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'train_loss', 'rec_loss', 'kl_loss', 'adv_loss', 
                          'val_loss', 'val_rec_loss', 'val_kl_loss', 'val_adv_loss']
            self.log_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            self.log_writer.writeheader()

    def train(self, num_epochs):
        self.vae.to(self.device)
        self.discriminator.to(self.device)
        for epoch in range(num_epochs):
            train_loss, rec_loss, kl_loss, adv_loss = self._train_epoch()
            val_loss, val_rec_loss, val_kl_loss, val_adv_loss = self._validate_epoch()
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Rec Loss: {rec_loss:.4f}, KL Loss: {kl_loss:.4f}, Adv Loss: {adv_loss:.4f}, Val Loss: {val_loss:.4f}, Val Rec Loss: {val_rec_loss:.4f}, Val KL Loss: {val_kl_loss:.4f}, Val Adv Loss: {val_adv_loss:.4f}')

            if (epoch + 1) % self.args.save_interval == 0:
                self._save_checkpoint(epoch + 1)

            self._log_losses(epoch + 1, train_loss, rec_loss, kl_loss, adv_loss, val_loss, val_rec_loss, val_kl_loss, val_adv_loss)

    def _train_epoch(self):
        self.vae.train()
        self.discriminator.train()
        total_loss, total_rec_loss, total_kl_loss, total_adv_loss = 0, 0, 0, 0
        for batch in self.train_loader:                        

            batch = batch.to(self.device)

            # Forward pass through VAE
            x_reconstructed, mu, logvar = self.vae(batch)

            # Reconstruction loss
            rec_loss = self.criterion(x_reconstructed, batch)

            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Adversarial loss
            real_labels = torch.ones(batch.size(0), 1).to(self.device)
            fake_labels = torch.zeros(batch.size(0), 1).to(self.device)

            real_outputs = self.discriminator(batch)
            x_reconstructed_detached = x_reconstructed.detach()
            fake_outputs = self.discriminator(x_reconstructed_detached)

            d_loss_real = F.binary_cross_entropy(real_outputs, real_labels)
            d_loss_fake = F.binary_cross_entropy(fake_outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            g_loss = F.binary_cross_entropy(fake_outputs, real_labels)

            # Total VAE loss
            vae_loss = self.lambda_rec * rec_loss + self.lambda_adv * g_loss + self.lambda_kl * kl_loss

            # Update VAE
            self.vae_optimizer.zero_grad()
            vae_loss.backward(retain_graph=True)
            self.vae_optimizer.step()

            # Update discriminator
            self.discriminator_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()


            total_loss += vae_loss.item()
            total_rec_loss += rec_loss.item()
            total_kl_loss += kl_loss.item()
            total_adv_loss += g_loss.item()
        
        return total_loss / len(self.train_loader), total_rec_loss / len(self.train_loader), total_kl_loss / len(self.train_loader), total_adv_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.vae.eval()
        self.discriminator.eval()
        total_val_loss, total_rec_loss, total_kl_loss, total_adv_loss = 0, 0, 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                x_reconstructed, mu, logvar = self.vae(batch)

                # Reconstruction loss
                rec_loss = self.criterion(x_reconstructed, batch)

                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # Adversarial loss
                real_labels = torch.ones(batch.size(0), 1).to(self.device)
                fake_labels = torch.zeros(batch.size(0), 1).to(self.device)

                real_outputs = self.discriminator(batch)
                x_reconstructed_detached = x_reconstructed.detach()
                fake_outputs = self.discriminator(x_reconstructed_detached)

                d_loss_real = F.binary_cross_entropy(real_outputs, real_labels)
                d_loss_fake = F.binary_cross_entropy(fake_outputs, fake_labels)
                d_loss = d_loss_real + d_loss_fake

                g_loss = F.binary_cross_entropy(fake_outputs, real_labels)

                # Total VAE loss
                vae_loss = self.lambda_rec * rec_loss + self.lambda_adv * g_loss + self.lambda_kl * kl_loss

                total_val_loss += vae_loss.item()
                total_rec_loss += rec_loss.item()
                total_kl_loss += kl_loss.item()
                total_adv_loss += g_loss.item()

        return total_val_loss / len(self.val_loader), total_rec_loss / len(self.val_loader), total_kl_loss / len(self.val_loader), total_adv_loss / len(self.val_loader)
    
    def _save_checkpoint(self, epoch):
        
        vae_path = os.path.join(self.args.ckpt_folder, f'vae_epoch_{epoch}.pth')
        discriminator_path = os.path.join(self.args.ckpt_folder, f'discriminator_epoch_{epoch}.pth')
        torch.save(self.vae.state_dict(), vae_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)
        print(f'Checkpoint saved for epoch {epoch}')

    def _log_losses(self, epoch, train_loss, rec_loss, kl_loss, adv_loss, val_loss, val_rec_loss, val_kl_loss, val_adv_loss):
        with open(self.args.logfile, 'a', newline='') as csvfile:
            self.log_writer = csv.writer(csvfile)
            self.log_writer.writerow([epoch, train_loss, rec_loss, kl_loss, adv_loss, val_loss, val_rec_loss, val_kl_loss, val_adv_loss])