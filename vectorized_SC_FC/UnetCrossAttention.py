import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])
    
    return embedding

class MyResNet1d(nn.Module):
    def __init__(self, L = 2278, in_c = 1, out_c = 1, kernel_size = 3, stride = 1, padding = 1, attn = False, attnheads = None, normalize=True):
        super(MyResNet1d, self).__init__()
        self.normalize = normalize
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
        out = self.ln1(x)  if self.normalize else x # shape (B, in_c, L) 
        out = self.activation(out) # (B, in_c, L)
        out = self.conv1(out) # (B, out_c, -)
        out = self.ln2(out) if self.normalize else out #(B, out_c, -)
        out = self.activation(out) #(B, out_c, -)
        out = self.conv2(out) #(B, out_c, -)
        #residual connection
        res_out = out + self.conv_input(x)

        if self.attn == True:
            out_attn = self.ln3(res_out) if self.normalize else res_out
            out_attn, _ = self.attention(res_out, res_out, res_out)
            res_out = res_out + out_attn
        return res_out
    
class CondUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100, in_channels=1, sequence_length= 512, cond_length=2278, attn = False, n_heads=8):
        super(CondUNet, self).__init__()
        
        self.n_steps = n_steps
        self.time_emb_dim = time_emb_dim
        self.L_lengths = [sequence_length, sequence_length // 2, (sequence_length // 2) // 2, ((sequence_length // 2) // 2)//2] 
        # Time embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.cond_emb = nn.Sequential(
            nn.Linear(cond_length, sequence_length),
            nn.ReLU(),
            nn.Linear(sequence_length, sequence_length),
            nn.SiLU(),
        )

        # SC Embedding
        self.cond_conv = nn.Conv1d(in_channels, in_channels, 3, 1, 1)
        self.cond_conv.bias.data.zero_()
        self.cond_conv.requires_grad = False #fix bias to zero to be able to not condition
        # Down Blocks

        ## First Down
        self.t_proj1 = self.time_proj(2 * in_channels)
        self.Resnet1 = nn.Sequential(
            MyResNet1d(L = sequence_length, in_c = 2 * in_channels, out_c = 4 * in_channels), #twice the channels because we concantenate condition information
            MyResNet1d(L = sequence_length, in_c = 4 * in_channels, out_c = 4 * in_channels),
            MyResNet1d(L = sequence_length, in_c = 4 * in_channels, out_c = 4 * in_channels, attn=attn, attnheads=n_heads)
        )
        self.down1 =  nn.Conv1d(4 * in_channels, 4 * in_channels, 4, 2, 1)

        ## Second Down
        self.t_proj2 = self.time_proj(4 * in_channels)
        self.Resnet2 = nn.Sequential(
            MyResNet1d(L = self.L_lengths[1], in_c = 4 * in_channels, out_c = 8 * in_channels), #twice the channels because we concantenate condition information
            MyResNet1d(L = self.L_lengths[1], in_c = 8 * in_channels, out_c = 8 * in_channels),
            MyResNet1d(L = self.L_lengths[1], in_c = 8 * in_channels, out_c = 8 * in_channels, attn=attn, attnheads=n_heads)
        )
        self.down2 =  nn.Conv1d(8 * in_channels, 8 * in_channels, 4, 2, 1)

        ## Third Down
        self.t_proj3 = self.time_proj(8 * in_channels)
        self.Resnet3 = nn.Sequential(
            MyResNet1d(L = self.L_lengths[2], in_c = 8 * in_channels, out_c = 16 * in_channels), #twice the channels because we concantenate condition information
            MyResNet1d(L = self.L_lengths[2], in_c = 16 * in_channels, out_c = 16 * in_channels),
            MyResNet1d(L = self.L_lengths[2], in_c = 16 * in_channels, out_c = 16 * in_channels, attn=attn, attnheads=n_heads)
        )
        self.down3 =  nn.Conv1d(16 * in_channels, 16 * in_channels, 4, 2, 1)

        # Mid
        self.t_proj_mid = self.time_proj(16 * in_channels)
        self.Resnet_mid = nn.Sequential(
            MyResNet1d(L = self.L_lengths[3], in_c = 16 * in_channels, out_c = 8 * in_channels), #twice the channels because we concantenate condition information
            MyResNet1d(L = self.L_lengths[3], in_c = 8 * in_channels, out_c = 8 * in_channels),
            MyResNet1d(L = self.L_lengths[3], in_c = 8 * in_channels, out_c = 16 * in_channels, attn=attn, attnheads=n_heads)
        )

        # Up 1
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(16*in_channels, 16*in_channels, kernel_size = 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(16*in_channels, 16*in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.t_proj_up1 = self.time_proj(32 * in_channels)
        self.ResnetUp1 = nn.Sequential(
            MyResNet1d(L = self.L_lengths[2], in_c = 32 * in_channels, out_c = 16 * in_channels), #twice the channels because we concantenate condition information
            MyResNet1d(L = self.L_lengths[2], in_c = 16 * in_channels, out_c = 8 * in_channels, attn=attn, attnheads=n_heads),
            MyResNet1d(L = self.L_lengths[2], in_c = 8 * in_channels, out_c = 8 * in_channels)
        )
        
        # Up 2
        self.up2 = nn.ConvTranspose1d(8 *in_channels, 8 *in_channels, 4, 2, 1)
        self.t_proj_up2 = self.time_proj(16 * in_channels)
        self.ResnetUp2 = nn.Sequential(
            MyResNet1d(L = self.L_lengths[1], in_c = 16 * in_channels, out_c = 8 * in_channels), #twice the channels because we concantenate condition information
            MyResNet1d(L = self.L_lengths[1], in_c = 8 * in_channels, out_c = 4 * in_channels),
            MyResNet1d(L = self.L_lengths[1], in_c = 4 * in_channels, out_c = 4 * in_channels, attn=attn, attnheads=n_heads)
        )

        #Up 3
        self.up3 = nn.ConvTranspose1d(4 *in_channels, 4 *in_channels, 4, 2, 1)
        self.t_proj_up3 = self.time_proj(8 * in_channels)
        self.ResnetUp3 = nn.Sequential(
            MyResNet1d(L = self.L_lengths[0], in_c = 8 * in_channels, out_c = 4 * in_channels), #twice the channels because we concantenate condition information
            MyResNet1d(L = self.L_lengths[0], in_c = 4 * in_channels, out_c = 2 * in_channels),
            MyResNet1d(L = self.L_lengths[0], in_c = 2 * in_channels, out_c =  2 * in_channels, normalize=False)
        )

        self.conv_out = nn.Conv1d(2 * in_channels, in_channels, 3, 1, 1)
    

    def forward(self, x, t, cond):
        #get the sinusoidal representation
        t = self.time_embed(t)
        batch_size = len(x)
        # Project condition to same dimention space
        cond_proj = self.cond_conv(self.cond_emb(cond)) 
        #input of NN is both condition and original
        joint_input = torch.cat((x, cond_proj), dim = 1) # concatenate on channels
        #Down
        Res_down1 = self.Resnet1(joint_input + self.t_proj1(t).reshape(batch_size, -1, 1))
        Res_down2 = self.Resnet2(self.down1(Res_down1) + self.t_proj2(t).reshape(batch_size, -1, 1))
        Res_down3 = self.Resnet3(self.down2(Res_down2) + self.t_proj3(t).reshape(batch_size, -1, 1))
        #Mid
        Res_mid = self.Resnet_mid(self.down3(Res_down3) + self.t_proj_mid(t).reshape(batch_size, -1, 1))
        # Skip connections and up

        augmented_down1 = torch.cat((Res_down3, self.up1(Res_mid)), dim = 1)
        augmented_down1 = self.ResnetUp1(augmented_down1 + self.t_proj_up1(t).reshape(batch_size, -1, 1))

        augmented_down2 = torch.cat((Res_down2, self.up2(augmented_down1)), dim = 1)
        augmented_down2 = self.ResnetUp2(augmented_down2 + self.t_proj_up2(t).reshape(batch_size, -1, 1))

        augmented_down3 = torch.cat((Res_down1, self.up3(augmented_down2)), dim = 1)
        augmented_down3 = self.ResnetUp3(augmented_down3 + self.t_proj_up3(t).reshape(batch_size, -1, 1))

        output = self.conv_out(augmented_down3)

        return output

    def time_proj(self, dim_out):
        return nn.Sequential(
            nn.Linear(self.time_emb_dim, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )


