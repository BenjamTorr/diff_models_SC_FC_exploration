import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])
    
    return embedding

class DiTBlockInContext(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, input_tokens, conditioning):
        """
        input_tokens: Tensor of shape (batch_size, seq_len, embed_dim)
        conditioning: Tensor of shape (batch_size, cond_len, embed_dim)
        """
        # Concatenate conditioning on sequence dimension
        x = torch.cat([input_tokens, conditioning], dim=1)  # (batch_size, seq_len, embed_dim + conditioning_dim)
        
        # Apply multi-head attention
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output  # Residual connection

        # Apply feedforward network
        x = self.norm2(x)
        ff_output = self.ffn(x)
        x = x + ff_output  # Residual connection

        return x[:, :input_tokens.size(1), :]



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlockAdaLNZero(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, emb_dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 6 * emb_dim, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        scaled_x1 = modulate(self.norm1(x), shift_msa, scale_msa) 
        x = x + gate_msa.unsqueeze(1) * self.attn(scaled_x1, scaled_x1, scaled_x1)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x 

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, emb_dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(emb_dim, patch_size * out_channels, bias=True) #reconstruct
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * emb_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiTNNadaLN(nn.Module):
    def __init__(self, channels_in = 1, L = 2278, patch_size = 7, emb_dim = 32, num_heads=5, mlp_ratio = 4.0,
                        n_steps = 1000, time_emb_dim = 100, cond_dim = 2278, num_blocks=12):
        """
        Finish this soon!!!!!!
        
        """
        super().__init__()
        self.patchify = PatchifyVect(channels_in = 1, L = L, patch_size=patch_size, embed_dim= emb_dim)
        #self.patchifyCond = PatchifyVect(channels_in = 1, L = 2278, patch_size=patch_size, embed_dim= emb_dim)
        self.L = L
        self.in_channels = channels_in
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_heads = num_heads,
        self.n_steps = n_steps
        self.time_emb_dim = time_emb_dim
        self.num_blocks = num_blocks
        self.cond_dim = cond_dim

        # Time and condition embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.te = self._make_te(time_emb_dim, self.emb_dim)
        self.ce = self._make_ce(cond_dim, emb_dim)
        

        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlockAdaLNZero(emb_dim = emb_dim, num_heads = num_heads, mlp_ratio= mlp_ratio) for _ in range(self.num_blocks)
        ])

        self.final_layer = FinalLayer(emb_dim=emb_dim, patch_size= self.patchify.patch_size, out_channels = channels_in)


        self.initialize_weights()

    def forward(self, x, t, condition):
        batch_size, in_channels, sequence_length = x.shape
        x = self.patchify(x)
        condition = self.ce(condition)
        t = self.time_embed(t)
        t = self.te(t).reshape(batch_size, 1, self.emb_dim)

        cond = t + condition
        cond = cond.reshape(batch_size, self.emb_dim)

        for block in self.blocks:
            x = block(x, cond)
        T = x.size(1)
        x = self.final_layer(x, cond) # (batch_size, T, patch_size)
        #reshape
        x = x.reshape(batch_size, in_channels, T * self.patchify.patch_size) #change if there is new channels
        
        #make sure we are reconstructing the same size
        assert x.size(2) == sequence_length, f"Wrong shape, x has shape {x.shape} and it should be ({batch_size} , {in_channels}, {sequence_length})"

        return x


    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
    
    def _make_ce(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
    

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        self.patchify.apply(_basic_init)
        #self.patchifyCond.apply(_basic_init)
        
        #patchify init
        nn.init.constant_(self.patchify.projection.bias, 0)

        #Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    
class PatchifyVect(nn.Module):
    def __init__(self, channels_in = 1, L = 2278, patch_size = 7, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_channels = channels_in
        self.L = L
        self.num_patches =self.L // self.patch_size

        self.projection = nn.Linear(self.in_channels * self.patch_size, embed_dim)

    def forward(self, x):
        batch_size, in_channels, L = x.shape #(32, 1, 2278)

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0,2,3,1,4).contiguous()
        patches = patches.view(batch_size,-1, self.patch_size)

        patch_embedding = self.projection(patches)

        return patch_embedding
        
class DiTNN(nn.Module):
    def __init__(self, channels_in = 1, L = 2278, patch_size = 7, emb_dim = 32, num_heads=5, ff_dim = 32,
                        n_steps = 1000, time_emb_dim = 100, cond_dim = 2278, num_blocks=12):
        """
        Finish this soon!!!!!!
        
        """
        super().__init__()
        self.patchify = PatchifyVect(channels_in = 1, L = L, patch_size=patch_size, embed_dim= emb_dim)
        #self.patchifyCond = PatchifyVect(channels_in = 1, L = 2278, patch_size=patch_size, embed_dim= emb_dim)
        self.L = L
        self.in_channels = channels_in
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_heads = num_heads,
        self.ff_dim = ff_dim
        self.n_steps = n_steps
        self.time_emb_dim = time_emb_dim
        self.num_blocks = num_blocks
        self.cond_dim = cond_dim

        # Time and condition embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.te = self._make_te(time_emb_dim, self.emb_dim)
        self.ce = self._make_ce(cond_dim, emb_dim)
        

        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlockInContext(embed_dim = emb_dim, num_heads = num_heads, ff_dim = ff_dim) for _ in range(self.num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(self.emb_dim)
        self.linear = nn.Linear(emb_dim, self.patchify.patch_size)

        self.initialize_weights()

    def forward(self, x, t, condition):
        batch_size, in_channels, sequence_length = x.shape
        x = self.patchify(x)
        condition = self.ce(condition)
        t = self.time_embed(t)
        t = self.te(t).reshape(batch_size, 1, self.emb_dim)

        cond = torch.cat([condition, t], dim = 1)

        for block in self.blocks:
            x = block(x, cond)
        x = self.layer_norm(x)
        x = self.linear(x)
        T = x.size(1)
        #reshape
        x = x.reshape(batch_size, in_channels, T * self.patchify.patch_size) # change if there is new channels
        
        #make sure we are reconstructing the same size
        assert x.size(2) == sequence_length, f"Wrong shape, x has shape {x.shape} and it should be ({batch_size} , {in_channels}, {sequence_length})"

        return x


    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
    
    def _make_ce(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
    

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        self.patchify.apply(_basic_init)
       # self.patchifyCond.apply(_basic_init)