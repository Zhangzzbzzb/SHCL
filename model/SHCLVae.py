# 使用patch- point两个mask进行重构，对比重构后的差异
from .embed import DataEmbedding, PositionalEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE Encoder
class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)  # Mean of latent space
        self.fc3 = nn.Linear(hidden_dim, latent_dim)  # Log variance of latent space
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

# VAE Decoder
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output dimension of original input
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_reconstructed = torch.sigmoid(self.fc2(h))  # Assuming input is normalized to [0, 1]
        return x_reconstructed

# VAE-based Encoder
class EncoderVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, norm_layer=None):
        super(EncoderVAE, self).__init__()
        self.vae_encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.norm = norm_layer
        self.vae_decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        
        mu, log_var = self.vae_encoder(x_flat)
        z = self.vae_encoder.reparameterize(mu, log_var)  # Sample from latent space
        
        # Reshape back to original sequence shape [B, T, D]
        z = z.view(B, T, -1)

        if self.norm is not None:
            z = self.norm(z)

        # Use the decoder to reconstruct the input
        x_reconstructed = self.vae_decoder(z)

        return x_reconstructed, (mu, log_var)

    
# usage of VAE-based Encoder
class SHCLVae(nn.Module):
    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, fr=0.4, tr=0.5, dev=None):
        super(SHCLVae, self).__init__()
        global device
        device = dev
        self.seq_size = seq_size  
        self.win_size = win_size
        self.enc_in = c_in
        self.d_model = d_model
        self.emb = DataEmbedding(c_in, d_model)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.seq_size // 2),
                                stride=1, padding=self.seq_size // 2, padding_mode="zeros", bias=False)

        # VAE-based Encoder
        self.single_encoder = EncoderVAE(
            input_dim=d_model,
            hidden_dim=256,   # Hidden layer dimension (tune this)
            latent_dim=d_model,   # Latent dimension (tune this)
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        self.pair_encoder = EncoderVAE(
            input_dim=d_model,
            hidden_dim=256,
            latent_dim=d_model,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        self.sp_pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
                
        self.sn_pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.pp_pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.pn_pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, L, M = x.shape  # (B, win_size, channel)

        # Mean normalization
        seq_mean = torch.mean(x, dim=1, keepdim=True)  # (B, 1, M)
        x = (x - seq_mean).to(device)  # Move x to GPU if available
        
        x = self.emb(x)  # (B, L, D)
        
        x = x.permute(0, 2, 1) # (B, M, L)
        x = self.conv1d(x.reshape(-1, 1, self.win_size)).reshape(-1, self.d_model, self.win_size) + x
        x = x.permute(0, 2, 1) # (B, L, M)   
        
        # Single-point mask on GPU
        mask_single_even = torch.arange(self.win_size, device=device) % 2 == 0
        mask_single_odd = ~mask_single_even
        
        xp_single, xn_single = x.clone(), x.clone()
        xp_single[:, ~mask_single_even, :] = 0  # Mask even positions
        xn_single[:, ~mask_single_odd, :] = 0  # Mask odd positions

        # Pair-point mask on GPU
        mask_pair_even = (torch.div(torch.arange(self.win_size, device=device), 2, rounding_mode='trunc') % 2) == 0
        mask_pair_odd = ~mask_pair_even
        
        xp_pair, xn_pair = x.clone(), x.clone()
        xp_pair[:, ~mask_pair_even, :] = 0  # Mask pair even positions
        xn_pair[:, ~mask_pair_odd, :] = 0  # Mask pair odd positions

        # Encoding
        xp_single_encoded, (mu_p_single, log_var_p_single) = self.single_encoder(xp_single)
        xn_single_encoded, (mu_n_single, log_var_n_single) = self.single_encoder(xn_single)

        xp_pair_encoded, (mu_p_pair, log_var_p_pair) = self.pair_encoder(xp_pair)
        xn_pair_encoded, (mu_n_pair, log_var_n_pair) = self.pair_encoder(xn_pair)

        xp_single_encoded = self.sp_pro(xp_single_encoded)
        xn_single_encoded = self.sn_pro(xn_single_encoded)
        
        xp_pair_encoded = self.pp_pro(xp_pair_encoded)
        xn_pair_encoded = self.pn_pro(xn_pair_encoded)
        
        return {
            "single": (xp_single_encoded, xn_single_encoded, mu_p_single, log_var_p_single, mu_n_single, log_var_n_single),
            "pair": (xp_pair_encoded, xn_pair_encoded, mu_p_pair, log_var_p_pair, mu_n_pair, log_var_n_pair)
        }