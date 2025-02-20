import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveScaling(nn.Module):
    """Adaptive layer normalization for correlation matrices, applied only to off-diagonal elements."""
    def __init__(self, input_dim, eps=1e-6):
        super(AdaptiveScaling, self).__init__()
        self.input_dim = input_dim
        self.eps = eps
        
        # Learnable parameters for adaptive transformation
        self.gamma = nn.Parameter(torch.ones(1))  # Scale
        self.beta = nn.Parameter(torch.zeros(1))  # Shift

    def forward(self, L_elements):
        """Apply adaptive normalization to the off-diagonal elements."""
        mean = L_elements.mean(dim=-1, keepdim=True)
        std = L_elements.std(dim=-1, keepdim=True) + self.eps  # Avoid division by zero
        
        # Normalize, then scale and shift
        L_normalized = (L_elements - mean) / std
        L_scaled = self.gamma * L_normalized + self.beta

        return torch.tanh(L_scaled)  # Keep values in (-1,1)

class CorrelationVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, device):
        super(CorrelationVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Encoder
        self.fc1 = nn.Linear(input_dim , hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        #adaptive scaling
        #self.adaptive_scaling = AdaptiveScaling(input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(self.device)
        eps = torch.randn_like(std).to(self.device)
        return (mu + eps * std).to(self.device)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        L_elements = self.fc4(h)  # Lower triangular elements
        #L_elements = self.adaptive_scaling(L_elements)
        return L_elements

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def sample(self, num_samples):
        """Generate new correlation matrices from the prior distribution."""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)  # Sample from N(0, I)
            rec = self.decode(z)
        return rec

# Loss function
def vae_loss(recon_x, x, mu, logvar, device):
    recon_loss = F.mse_loss(recon_x, x).to(device)
    kl_div = -1e-6 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, device):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        latent_x = self.encoder(x)
        recon_x = self.decoder(latent_x)
        return recon_x
########################################## Advanced Model, experimental #############################
#####################################################################################################

class MyResNet1d(nn.Module):
    def __init__(self, L = 2278, in_c = 1, out_c = 1, kernel_size = 3, stride = 1, padding = 1, attn = False, attnheads = None):
        super(MyResNet1d, self).__init__()
        self.attn = attn
        self.attnheads = attnheads
        self.ln1 = nn.LayerNorm(normalized_shape = (in_c, L))
        self.ln2 = nn.LayerNorm(normalized_shape = (out_c, L))
        self.ln3= nn.LayerNorm(normalized_shape = (out_c, L)) if self.attn else None
        self.attention = nn.MultiheadAttention(L, attnheads, batch_first = True) if self.attn == True else None
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size, stride, padding)
        self.conv_input = nn.Conv1d(in_c, out_c, kernel_size = 1)
        self.activation = nn.SiLU()

    def forward(self, x):
        out = self.ln1(x)  # shape (B, in_c, L)
        out = self.activation(out) # (B, in_c, L)
        out = self.conv1(out) # (B, out_c, -)
        out = self.ln2(out) #(B, out_c, -)
        out = self.activation(out) #(B, out_c, -)
        out = self.conv2(out) #(B, out_c, -)
        #residual connection
        res_out = out + self.conv_input(x)

        if self.attn == True:
            out_attn = self.ln3(res_out)
            out_attn, _ = self.attention(res_out, res_out, res_out)
            res_out = res_out + out_attn
        return res_out






         
class AdvancedAutoEncoder(nn.Module):
    def __init__(self, in_channels = 1, L = 2304, VAE=False):
        super(AdvancedAutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.VAE = VAE
        self.out_channels = 2 * in_channels if VAE else in_channels 
        self.L = L
        self.L_sizes = [L, L // 2, (L // 2) // 2]

        # Down 
        self.Resnet1 = nn.Sequential(
            MyResNet1d(L, in_channels, self.out_channels),
            MyResNet1d(L, self.out_channels, self.out_channels),
            MyResNet1d(L, self.out_channels, self.out_channels)
        )
        self.down1 = nn.Conv1d(self.out_channels, self.out_channels, 4, 2, 1) 

        self.Resnet2 = nn.Sequential(
            MyResNet1d(self.L_sizes[1], self.out_channels, self.out_channels),
            MyResNet1d(self.L_sizes[1], self.out_channels, self.out_channels),
            MyResNet1d(self.L_sizes[1], self.out_channels, self.out_channels)
        )

        self.down2 = nn.Conv1d(self.out_channels, self.out_channels, 4, 2, 1) 

        #Mid
        self.middle = nn.Sequential(
            MyResNet1d(self.L_sizes[2], in_c = self.out_channels, out_c = self.out_channels),
            MyResNet1d(self.L_sizes[2], self.out_channels, self.out_channels),
            MyResNet1d(self.L_sizes[2], self.out_channels, self.out_channels)
        )

        #up


        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose1d(in_channels, in_channels, 2, 1)
        )

        self.Resnet3 = nn.Sequential(
            MyResNet1d(self.L_sizes[1], in_channels, in_channels),
            MyResNet1d(self.L_sizes[1], in_channels, in_channels),
            MyResNet1d(self.L_sizes[1], in_channels, in_channels)
        )
        
        if self.L_sizes[0] != (self.L_sizes[1] * 2):
            output_padding1 = 1
        else:
            output_padding1 = 0
        

        self.up2 = nn.ConvTranspose1d(in_channels, in_channels, 4, 2, 1, output_padding=output_padding1)

        self.Resnet4 = nn.Sequential(
            MyResNet1d(self.L_sizes[0], in_channels, in_channels),
            MyResNet1d(self.L_sizes[0], in_channels, in_channels),
            MyResNet1d(self.L_sizes[0], in_channels, in_channels)
        )
        
        self.last_activation = nn.Tanh()

    def encoder(self, x):
        x = self.Resnet1(x) #(B, C_out, L 2)
        x = self.down1(x) # (B, C_out, (L // 2))
        x = self.Resnet2(x) # (B, C_out, (L // 2))
        x = self.down2(x) # (B, C_out, (L // 2) // 2)
        x = self.middle(x) # (B, C_out, (L // 2) // 2)

        if self.VAE:
            mu = x[:, :self.in_channels, :]
            logvar = x[:, self.in_channels:, :]
            z = self.reparameterize(mu, logvar) # (B, in_channels, latent_dim)
            return z, mu, logvar
        else:
            z = x # (B, in_channels, latent_dim)
            mu = None
            logvar = None
            return z

    def decoder(self, x):
        x = self.up1(x)
        x = self.Resnet3(x)
        x = self.up2(x)
        x = self.Resnet4(x)
        x = self.last_activation(x)
        return x
    
    def forward(self, x):
        if self.VAE:
            latent_x, mu, logvar = self.encoder(x)
            recon_x = self.decoder(latent_x)
            return recon_x, mu, logvar
        else:
            latent_x = self.encoder(x)
            recon_x = self.decoder(latent_x)
        return recon_x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std)

