import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import VAE, Discriminator
from trainer import Trainer
import os
import numpy as np

def main(args):

    train_dataset = CustomDataset(args.data_dir, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = CustomDataset(args.val_data_dir, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Switching to CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    print('Using device: ', device)

    vae = VAE(image_channels=1, latent_dim=args.latent_dim).to(device)
    
    discriminator = Discriminator(image_channels=1).to(device)

    vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    trainer = Trainer(vae, 
                      discriminator, 
                      train_loader, 
                      val_loader, 
                      criterion, 
                      vae_optimizer, 
                      discriminator_optimizer, 
                      args=args)
    trainer.train(num_epochs=40)


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        image = np.load(file_path)  # Load the .npy file
        image = torch.tensor(image, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)

        return image

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train an adversarial VAE ')
    parser.add_argument('--data_dir', type=str, default='data_train', help='Path to training data')
    parser.add_argument('--val_data_dir', type=str, default='data_val', help='Path to validation data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--logfile', type=str, default='training_logs.csv', help='Log file')
    parser.add_argument('--lambda_rec', type=float, default=1, help='Reconstruction loss weight')
    parser.add_argument('--lambda_kl', type=float, default=1, help='KL divergence loss weight')
    parser.add_argument('--lambda_adv', type=float, default=1, help='Adversarial loss weight')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
    parser.add_argument('--ckpt_folder', type=str, default='checkpoints', help='Folder to save checkpoints' )
    parser.add_argument('--save_interval', type=int, default=5, help='Interval to save checkpoints')
    parser.add_argument('--latent_dim', type=int, default=512, help='Dimension of latent space')
    args = parser.parse_args()
    main(args)