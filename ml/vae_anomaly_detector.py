"""
Variational Autoencoder for market anomaly detection.
Extracted from neural_network_prediction_engine.py — standalone, minimal.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class VariationalAutoEncoder(nn.Module):
    """VAE for anomaly detection. Learns to reconstruct 'normal' market
    feature vectors; high reconstruction error → anomalous regime."""

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Optional[List[int]] = None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            ])
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            ])
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon: torch.Tensor, x: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE ELBO loss = reconstruction + KL divergence."""
        recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss
