import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union

class FraudVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for fraud and anomaly detection in sequential data.

    This VAE processes sequences by first flattening them, then passing them through
    an MLP-based encoder to get latent variable parameters (mu and log_var).
    A latent vector z is sampled using the reparameterization trick.
    An MLP-based decoder then attempts to reconstruct the original flattened sequence from z.

    Key Features:
    - Handles sequences of `seq_len` timesteps, each with `input_dim` features.
    - Uses a configurable latent dimension (`latent_dim`).
    - Supports different reconstruction loss modes: 'mse', 'mae', 'cosine'.
    - Allows for trainable `feature_weights` to emphasize certain input features
      during reconstruction for 'mse' and 'mae' modes.
    - Implements the standard VAE loss (reconstruction + KL divergence).
    - Includes a `beta` parameter (for Beta-VAE) to control the weight of the KL divergence term.
    - Provides an `anomaly_score` method based on reconstruction error.

    Note on "Temporal Attention": The original docstring mentioned temporal attention.
    This current MLP-based architecture does not explicitly implement mechanisms like
    LSTMs, GRUs, or Transformer attention layers for capturing temporal dependencies
    *within* the VAE's encoder/decoder MLPs themselves. The model processes the
    entire sequence once flattened. Temporal patterns are implicitly learned if
    present in the overall structure of the flattened sequence.
    """

    def __init__(self, 
                 input_dim: int, 
                 seq_len: int, 
                 latent_dim: int = 64, 
                 recon_mode: str = "mse",
                 beta: float = 1.0):
        """
        Initializes the FraudVAE model.

        Args:
            input_dim (int): The number of features for each element in the sequence.
            seq_len (int): The length of the input sequences.
            latent_dim (int, optional): The dimensionality of the latent space. Defaults to 64.
            recon_mode (str, optional): The reconstruction loss mode.
                Options: 'mse', 'mae', 'cosine'. Defaults to "mse".
            beta (float, optional): The weight for the KL divergence term (for Beta-VAE).
                Defaults to 1.0 (standard VAE).
        """
        super(FraudVAE, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.flattened_dim = input_dim * seq_len
        self.recon_mode = recon_mode.lower()
        self.beta = beta

        if self.recon_mode not in ["mse", "mae", "cosine"]:
            raise ValueError(f"Unsupported recon_mode: {recon_mode}. Choose from 'mse', 'mae', 'cosine'.")

        # === Encoder ===
        # Input: Flattened sequence [Batch, flattened_dim]
        # Output: Parameters for latent distribution (mu and log_var)
        self.encoder = nn.Sequential(
            nn.Linear(self.flattened_dim, 256),
            nn.BatchNorm1d(256), # Normalizes over the 256 features
            nn.ReLU(),
            nn.Dropout(p=0.2), # Added dropout for regularization
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2), # Added dropout for regularization
            nn.Linear(128, latent_dim * 2)  # For mu and log_var
        )

        # === Decoder ===
        # Input: Latent vector z [Batch, latent_dim]
        # Output: Reconstructed flattened sequence [Batch, flattened_dim]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.flattened_dim),
            # Sigmoid assumes input features are scaled to [0, 1].
            # If features have a different range (e.g., standardized),
            # this might need to be removed or changed (e.g., to Tanh for [-1,1]).
            nn.Sigmoid()
        )

        # Trainable weights for feature importance in reconstruction (for mse/mae)
        # Shape: [input_dim] - applied per feature across all timesteps.
        self.feature_weights = nn.Parameter(torch.ones(input_dim))

    def encode(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the flattened input into mu and log_var of the latent space."""
        # x_flat shape: [Batch, flattened_dim]
        stats = self.encoder(x_flat)
        mu, log_var = torch.chunk(stats, 2, dim=-1) # Split along the last dimension
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Performs the reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # Samples from N(0, I)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes the latent vector z into a reconstructed flattened sequence."""
        # z shape: [Batch, latent_dim]
        # Output shape: [Batch, flattened_dim]
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs a full forward pass of the VAE.

        Args:
            x (torch.Tensor): Input sequence tensor of shape [Batch, seq_len, input_dim].

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                'x_flat': The input tensor flattened to [Batch, flattened_dim].
                'mu': The mean of the latent distribution, shape [Batch, latent_dim].
                'log_var': The log variance of the latent distribution, shape [Batch, latent_dim].
                'z': The sampled latent vector, shape [Batch, latent_dim].
                'recon_flat': The reconstructed sequence, flattened, shape [Batch, flattened_dim].
                'recon_seq': The reconstructed sequence, reshaped to [Batch, seq_len, input_dim].
        """
        # x shape: [Batch, seq_len, input_dim]
        x_flat = x.view(x.size(0), -1)  # Shape: [Batch, flattened_dim]

        mu, log_var = self.encode(x_flat)
        z = self.reparameterize(mu, log_var)
        recon_flat = self.decode(z)  # Shape: [Batch, flattened_dim]
        
        # Reshape reconstructed output to sequence form for easier comparison or use
        recon_seq = recon_flat.view(x.size(0), self.seq_len, self.input_dim)

        return {
            "x_flat": x_flat,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "recon_flat": recon_flat,
            "recon_seq": recon_seq
        }

    def reconstruction_loss(self, recon_flat: torch.Tensor, target_flat: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reconstruction loss.

        Args:
            recon_flat (torch.Tensor): The flattened reconstructed sequence from the decoder.
                                       Shape: [Batch, flattened_dim].
            target_flat (torch.Tensor): The original flattened input sequence.
                                        Shape: [Batch, flattened_dim].
        Returns:
            torch.Tensor: The scalar reconstruction loss.
        """
        B = target_flat.size(0)

        if self.recon_mode in ["mse", "mae"]:
            # Reshape to apply feature_weights per original feature dimension
            recon_seq = recon_flat.view(B, self.seq_len, self.input_dim)
            target_seq = target_flat.view(B, self.seq_len, self.input_dim)
            diff = recon_seq - target_seq
            # self.feature_weights has shape [input_dim]
            weighted_diff = diff * self.feature_weights # Broadcasting

            if self.recon_mode == "mse":
                loss = (weighted_diff ** 2).mean()
            else: # mae
                loss = weighted_diff.abs().mean()
        elif self.recon_mode == "cosine":
            # Cosine similarity is calculated on the unweighted flattened vectors.
            loss = 1 - F.cosine_similarity(recon_flat, target_flat, dim=-1).mean()
        else:
            # This case should ideally not be reached due to init check, but as a safeguard:
            raise ValueError(f"Unsupported recon_mode encountered in reconstruction_loss: {self.recon_mode}")
            
        return loss

    def vae_loss(self, x_orig_seq: torch.Tensor, model_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculates the total VAE loss (reconstruction + beta * KL divergence).

        Args:
            x_orig_seq (torch.Tensor): The original input sequence tensor.
                                       Shape: [Batch, seq_len, input_dim]. (Not directly used if x_flat is in model_output)
            model_output (Dict[str, torch.Tensor]): The dictionary returned by the forward pass.
                                                    Must contain 'recon_flat', 'x_flat', 'mu', 'log_var'.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                'total_loss': The combined VAE loss.
                'recon_loss': The reconstruction loss component.
                'kld_loss': The KL divergence component.
        """
        recon_loss_val = self.reconstruction_loss(model_output["recon_flat"], model_output["x_flat"])
        
        mu = model_output["mu"]
        log_var = model_output["log_var"]
        
        # KL divergence: D_KL(Q(z|X) || P(z))
        # P(z) is N(0, I)
        # Sum over latent dimensions, then mean over batch
        kld_val = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        total_loss_val = recon_loss_val + self.beta * kld_val
        
        return {
            "total_loss": total_loss_val,
            "recon_loss": recon_loss_val,
            "kld_loss": kld_val
        }

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes an anomaly score for the input sequence(s) based on reconstruction error.
        Higher scores indicate higher anomaly.

        Args:
            x (torch.Tensor): Input sequence tensor of shape [Batch, seq_len, input_dim].

        Returns:
            torch.Tensor: Anomaly scores for each sample in the batch, shape [Batch].
        """
        self.eval()  # Ensure model is in evaluation mode (e.g., for dropout)
        with torch.no_grad():
            model_output = self.forward(x) # Get reconstructions and original flattened input
            recon_flat = model_output["recon_flat"]
            x_flat = model_output["x_flat"]
            
            B = x_flat.size(0)

            if self.recon_mode in ["mse", "mae"]:
                recon_seq = recon_flat.view(B, self.seq_len, self.input_dim)
                target_seq = x_flat.view(B, self.seq_len, self.input_dim)
                diff_seq = recon_seq - target_seq
                weighted_diff_seq = diff_seq * self.feature_weights # Apply feature weights

                if self.recon_mode == "mse":
                    error = (weighted_diff_seq ** 2).mean(dim=[1, 2]) # Mean over seq_len and input_dim
                else: # mae
                    error = weighted_diff_seq.abs().mean(dim=[1, 2])
            elif self.recon_mode == "cosine":
                # Higher score for less similar (more anomalous)
                error = 1 - F.cosine_similarity(recon_flat, x_flat, dim=-1)
            else:
                # Fallback, though init should prevent this
                error = F.mse_loss(recon_flat, x_flat, reduction='none').mean(dim=1)
        
        return error # Shape: [Batch]

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Configuration
    batch_s = 4
    seq_l = 10  # Sequence length
    feat_dim = 5 # Feature dimension
    lat_dim = 3  # Latent dimension
    beta_val = 0.5

    # Create dummy data
    dummy_data = torch.rand(batch_s, seq_l, feat_dim) # Assumes data is in [0,1] for Sigmoid

    # Initialize model
    vae_model = FraudVAE(input_dim=feat_dim, seq_len=seq_l, latent_dim=lat_dim, recon_mode="mse", beta=beta_val)
    
    # Forward pass
    output = vae_model(dummy_data)
    print("Forward pass output keys:", output.keys())
    print("x_flat shape:", output["x_flat"].shape)
    print("recon_flat shape:", output["recon_flat"].shape)
    print("recon_seq shape:", output["recon_seq"].shape)
    print("mu shape:", output["mu"].shape)
    print("z shape:", output["z"].shape)

    # Calculate loss
    loss_dict = vae_model.vae_loss(dummy_data, output)
    print(f"Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Reconstruction Loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"KLD Loss (beta-weighted): {(vae_model.beta * loss_dict['kld_loss']).item():.4f} (raw KLD: {loss_dict['kld_loss'].item():.4f})")

    # Calculate anomaly scores
    anomaly_scores = vae_model.anomaly_score(dummy_data)
    print("Anomaly scores shape:", anomaly_scores.shape)
    print("Anomaly scores:", anomaly_scores)

    # Test with cosine
    vae_model_cosine = FraudVAE(input_dim=feat_dim, seq_len=seq_l, latent_dim=lat_dim, recon_mode="cosine")
    output_cosine = vae_model_cosine(dummy_data)
    loss_dict_cosine = vae_model_cosine.vae_loss(dummy_data, output_cosine)
    print(f"\nCosine Total Loss: {loss_dict_cosine['total_loss'].item():.4f}")
    anomaly_scores_cosine = vae_model_cosine.anomaly_score(dummy_data)
    print("Cosine Anomaly scores:", anomaly_scores_cosine)
