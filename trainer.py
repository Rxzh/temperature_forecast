import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device='cpu', lambda_rec=1, lambda_adv=1, lambda_kl=1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.lambda_rec = lambda_rec
        self.lambda_adv = lambda_adv
        self.lambda_kl = lambda_kl

    def train(self, num_epochs):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            train_loss, rec_loss, kl_loss, adv_loss = self._train_epoch()
            val_loss = self._validate_epoch()
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Rec Loss: {rec_loss:.4f}, KL Loss: {kl_loss:.4f}, Adv Loss: {adv_loss:.4f}, Val Loss: {val_loss:.4f}')

    def _train_epoch(self):
        self.model.train()
        total_loss, total_rec_loss, total_kl_loss, total_adv_loss = 0, 0, 0, 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            vae_loss, rec_loss, kl_loss, adv_loss = self.model._run_epoch(batch, self.optimizer, self.criterion, self.lambda_rec, self.lambda_adv, self.lambda_kl)
            total_loss += vae_loss
            total_rec_loss += rec_loss
            total_kl_loss += kl_loss
            total_adv_loss += adv_loss
        return total_loss / len(self.train_loader), total_rec_loss / len(self.train_loader), total_kl_loss / len(self.train_loader), total_adv_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                x_reconstructed, mu, logvar = self.model(batch)
                rec_loss = self.criterion(x_reconstructed, batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss = rec_loss + kl_loss
                total_val_loss += val_loss.item()
        return total_val_loss / len(self.val_loader)

