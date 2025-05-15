import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class FraudGenerator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()  # Assumes normalized [-1, 1] inputs
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FraudDiscriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FraudGANSynthesizer:
    """
    GAN-based fraud sample generator for minority class augmentation.

    Usage:
        synth = FraudGANSynthesizer(input_dim=16)
        synth.train(real_fraud_tensor)
        synthetic = synth.generate(n_samples=100)
    """

    def __init__(self, input_dim: int = 16, latent_dim: int = 32, device: str = "cpu"):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        self.generator = FraudGenerator(latent_dim, input_dim).to(self.device)
        self.discriminator = FraudDiscriminator(input_dim).to(self.device)

        self.criterion = nn.BCELoss()
        self.gen_opt = optim.Adam(self.generator.parameters(), lr=1e-3)
        self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=1e-3)

    def train(self, real_data: torch.Tensor, epochs: int = 1000, batch_size: int = 64) -> None:
        real_data = real_data.to(self.device)

        for epoch in range(epochs):
            idx = torch.randint(0, real_data.size(0), (batch_size,))
            real_batch = real_data[idx]

            # === Train Discriminator ===
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_batch = self.generator(z)

            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)

            real_preds = self.discriminator(real_batch)
            fake_preds = self.discriminator(fake_batch.detach())

            d_loss = self.criterion(real_preds, real_labels) + self.criterion(fake_preds, fake_labels)
            self.disc_opt.zero_grad()
            d_loss.backward()
            self.disc_opt.step()

            # === Train Generator ===
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            gen_data = self.generator(z)
            gen_preds = self.discriminator(gen_data)

            g_loss = self.criterion(gen_preds, real_labels)  # Fool discriminator
            self.gen_opt.zero_grad()
            g_loss.backward()
            self.gen_opt.step()

            if (epoch + 1) % 200 == 0:
                print(f"[{epoch+1}/{epochs}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    def generate(self, n_samples: int = 100) -> torch.Tensor:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(z)
        return samples.cpu()
