import torch
import numpy as np
import torch.nn as nn 
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(42)

BATCH_SIZE = 128
NUM_EPOCHS = 30
DISPLAY_LIMIT = 3
EVAL_INTERVAL = 1
DISPLAY_DURING_EVAL  = True
PATIENCE = 5
Z_DIM = 32
FREE_BITS = 0.05
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

################################ Dataset ####################################################
class ImageDataset(Dataset):
    def __init__(self, npy_path: str):
        # Load list of episodes (each episode is [steps, h, w, c])
        self.data_array = np.load(npy_path, allow_pickle=True)
        # Compute cumulative lengths to index into flattened dataset
        lengths = [episode.shape[0] for episode in self.data_array]
        self.cum_lengths = np.cumsum(lengths)

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, global_idx: int):
        # Find which episode this index belongs to
        episode_idx = np.searchsorted(self.cum_lengths, global_idx, side='right')
        start = 0 if episode_idx == 0 else self.cum_lengths[episode_idx - 1]
        local_idx = global_idx - start

        image = self.data_array[episode_idx][local_idx].astype('float32') / 255.0
        image = torch.tensor(image).permute(2, 0, 1)
        return image

################################ VAE Model ####################################################
class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 96 -> 48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),             # 48 -> 24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),            # 24 -> 12
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),           # 12 -> 6
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),           # 6 -> 6
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.mu = nn.Linear(512*6*6, latent_dim)
        self.logvar = nn.Linear(512*6*6, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mu = self.mu(x) * 0.1
        logvar = self.logvar(x) * 0.1
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=32, output_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512*6*6)
        self.unflatten = nn.Unflatten(1, (512, 6, 6))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),  # 6 -> 6
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 6 -> 12
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 12 -> 24
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 24 -> 48
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # 48 -> 96
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        x = self.deconv(x)
        return x

class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=Z_DIM):
        super().__init__()
        self.encoder = Encoder(image_channels, latent_dim)
        self.decoder = Decoder(latent_dim, image_channels)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
class VAELoss(nn.Module):
    def __init__(self, r_loss_factor=1.0, beta=0.1, free_bits=0.05, kl_tolerence = 0.5):
        """
        free_bits: minimum KL contribution per latent dimension
        beta: weight for KL term
        r_loss_factor: scales reconstruction loss
        """
        super().__init__()
        self.r_loss_factor = r_loss_factor
        self.beta = beta
        self.free_bits = free_bits
        self.mse_loss = nn.MSELoss()
        self.kl_tolerance = 0.5
        self.z_size = 32

    def forward(self, input_image, decoder_output, mus, logvars, epoch= None):
        # Reconstruction Loss (MSE but summed over pixels per sample)
        recon_loss = torch.sum((decoder_output - input_image) ** 2, dim=(1,2,3))
        recon_loss = torch.mean(recon_loss)

        # KL loss like TF
        kl_loss = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp(), dim=1)
        kl_loss = torch.clamp(kl_loss, min=self.kl_tolerance * self.z_size)
        kl_loss = torch.mean(kl_loss)

        loss = recon_loss + kl_loss
        return loss
    
    # def forward(self, input_image, decoder_output, mus, logvars, epoch=None):
    #     # Reconstruction loss
    #     recon_loss = self.mse_loss(decoder_output, input_image) * self.r_loss_factor

    #     # KL per latent dimension
    #     kl_per_dim = -0.5 * (1 + logvars - mus.pow(2) - logvars.exp())  # [batch, latent_dim]

    #     # Clamp each dimension to free_bits
    #     kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)

    #     # Sum over dimensions, then mean over batch
    #     kl_loss = kl_per_dim.sum(dim=1).mean()

    #     # Total loss
    #     return recon_loss + self.beta * kl_loss

################################ Utils ####################################################
def display_image(autoencoder, test_dataloader, limit = None):
    limit = DISPLAY_LIMIT if limit is None else limit
    with torch.no_grad():  # note the parentheses
        autoencoder.eval()
        inputs = next(iter(test_dataloader)).to(DEVICE)
        outputs  = autoencoder(inputs)[0]
        rands = torch.rand((outputs.shape[0], 32)).to(DEVICE)
        rands_out = autoencoder.decoder(rands)

        orig = inputs[:limit].cpu()
        recon = outputs[:limit].cpu()
        rands_out = rands_out[:limit].cpu()

        orig_np = orig.permute(0,2,3,1).numpy()
        recon_np = recon.permute(0,2,3,1).numpy()
        rands_np = rands_out.permute(0,2,3,1).numpy()

        fig, axes = plt.subplots(limit, 3, figsize=(6, 3*limit))
        for i in range(limit):
            # Original image
            if limit  > 1:
                axes[i, 0].imshow(orig_np[i])
                axes[i, 0].set_title("Original")
                axes[i, 0].axis("off")

                # Reconstructed image
                axes[i, 1].imshow(recon_np[i])
                axes[i, 1].set_title("Reconstructed")
                axes[i, 1].axis("off")

                axes[i, 2].imshow(rands_np[i])
                axes[i, 2].set_title("Random")
                axes[i, 2].axis("off")
            else:
                axes[0].imshow(orig_np[i])
                axes[0].set_title("Original")
                axes[0].axis("off")

                axes[1].imshow(recon_np[i])
                axes[1].set_title("Reconstructed")
                axes[1].axis("off")

                axes[2].imshow(rands_np[i])
                axes[2].set_title("Random")
                axes[2].axis("off")


        plt.tight_layout()
        plt.show()

def evaluate(model, dataloader, loss_fn, epoch):
    avg_loss = 0
    model.eval()
    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            inputs = batch.to(DEVICE)
            outputs, mus, logvars = model(inputs)
            loss = loss_fn(outputs, inputs, mus, logvars, epoch)
            avg_loss += loss
    avg_loss = avg_loss/ (len(dataloader))

    if DISPLAY_DURING_EVAL:
        display_image(model, dataloader, 1)

    return avg_loss.item()

def display_single(model: nn.Module, vector: torch.Tensor = None):
    model.eval()
    with torch.no_grad():
        if vector == None:
            vector = torch.rand((1,32))
        
        vector = vector.to(DEVICE)
        model = model.to(DEVICE)
        output = model(vector)
        recon = output.cpu().permute(0,2,3,1).numpy()

        fig, axes = plt.subplots(1, 1, figsize=(6, 3))
            
        axes.imshow(recon[0])
        axes.set_title("Reconstruction")
        axes.axis("off")


        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test = ImageDataset('../data/600k_rollouts.npy')
    print(len(test))