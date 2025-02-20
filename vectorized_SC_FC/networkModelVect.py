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


class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out
    
class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100, feat_dim = 1):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.feat1 = nn.Sequential(
            MyBlock((1, 2278), 1, 10),
            MyBlock((10, 2278), 10, 10),
            MyBlock((10, 2278), 10, 10)
        )
        
        self.b1 = nn.Sequential(
            MyBlock((1, 2278), 1, 10),
            MyBlock((10, 2278), 10, 10),
            MyBlock((10, 2278), 10, 10)
        )
        self.down1 =     nn.Conv1d(10, 10, 4, 2, 1)
        self.downcond1 = nn.Conv1d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.feat2 = nn.Sequential(
            MyBlock((10, 1139), 10, 20),
            MyBlock((20, 1139), 20, 20),
            MyBlock((20, 1139), 20, 20)
        )

        self.b2 = nn.Sequential(
            MyBlock((10, 1139), 10, 20),
            MyBlock((20, 1139), 20, 20),
            MyBlock((20, 1139), 20, 20)
        )
        self.down2 = nn.Conv1d(20, 20, 4, 2, 1)
        self.downcond2 = nn.Conv1d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.feat3 = nn.Sequential(
            MyBlock((20, 569), 20, 40),
            MyBlock((40, 569), 40, 40),
            MyBlock((40, 569), 40, 40)
        )

        self.b3 = nn.Sequential(
            MyBlock((20, 569), 20, 40),
            MyBlock((40, 569), 40, 40),
            MyBlock((40, 569), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv1d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv1d(40, 40, 4, 2, 1)
        )

        self.downcond3 = nn.Sequential(
            nn.Conv1d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv1d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.feat_mid = nn.Sequential(
            MyBlock((40, 284), 40, 20),
            MyBlock((20, 284), 20, 20),
            MyBlock((20, 284), 20, 40)
        )

        self.b_mid = nn.Sequential(
            MyBlock((40, 284), 40, 20),
            MyBlock((20, 284), 20, 20),
            MyBlock((20, 284), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose1d(40, 40, 2, 1)
        )

        self.upcond1 = nn.Sequential(
            nn.ConvTranspose1d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose1d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.feat4 = nn.Sequential(
            MyBlock((80, 569), 80, 40),
            MyBlock((40, 569), 40, 20),
            MyBlock((20, 569), 20, 20)
        )
        
        self.b4 = nn.Sequential(
            MyBlock((80, 569), 80, 40),
            MyBlock((40, 569), 40, 20),
            MyBlock((20, 569), 20, 20)
        )

        self.up2 = nn.ConvTranspose1d(20, 20, 4, 2, 1, output_padding=1)
        self.upcond2 = nn.ConvTranspose1d(20, 20, 4, 2, 1, output_padding=1)

        self.te5 = self._make_te(time_emb_dim, 40)
        self.feat5 = nn.Sequential(
            MyBlock((40, 1139), 40, 20),
            MyBlock((20, 1139), 20, 10),
            MyBlock((10, 1139), 10, 10)
        )

        self.b5 = nn.Sequential(
            MyBlock((40, 1139), 40, 20),
            MyBlock((20, 1139), 20, 10),
            MyBlock((10, 1139), 10, 10)
        )

        self.up3 = nn.ConvTranspose1d(10, 10, 4, 2, 1)
        self.upcond3 = nn.ConvTranspose1d(10, 10, 4, 2, 1)
        
        self.te_out = self._make_te(time_emb_dim, 20)
        self.feat_out = nn.Sequential(
            MyBlock((20, 2278), 20, 10),
            MyBlock((10, 2278), 10, 10),
            MyBlock((10, 2278), 10, 10, normalize=False)
        )
        
        self.b_out = nn.Sequential(
            MyBlock((20, 2278), 20, 10),
            MyBlock((10, 2278), 10, 10),
            MyBlock((10, 2278), 10, 10, normalize=False)
        )


        self.conv_out = nn.Conv1d(10, 1, 3, 1, 1)

    def forward(self, x, t, cond_info):

        t = self.time_embed(t)
        n = len(x)
        assert x.shape == cond_info.shape

        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1))  # (N, 10, 28, 28)
        out_cond1 = self.feat1(cond_info)
        down_cond1 = self.downcond1(out_cond1)
        
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1) + down_cond1)  # (N, 20, 14, 14)
        out_cond2 = self.feat2(down_cond1)
        down_cond2 = self.downcond2(out_cond2)

        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1) + down_cond2)  # (N, 40, 7, 7)
        out_cond3 = self.feat3(down_cond2)
        down_cond3 = self.downcond3(out_cond3)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1) + down_cond3)  # (N, 40, 3, 3)
        out_condmid = self.feat_mid(down_cond3)
        up_cond1 = self.upcond1(out_condmid)


        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out_cond4 = torch.cat((out_cond3, up_cond1), dim=1)
        
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1) + out_cond4)  # (N, 20, 7, 7)
        out_cond4 = self.feat4(out_cond4)
        up_cond2 = self.upcond2(out_cond4)

        out_cond5 = torch.cat((out_cond2, up_cond2), dim=1)
        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1) + out_cond5)  # (N, 10, 14, 14)
        out_cond5 = self.feat5(out_cond5)
        
        out_up3 = self.upcond3(out_cond5)
        out_cond = torch.cat((out_cond1, out_up3), dim=1)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1) + out_cond)  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_fe(self, dim_in, dim_out, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels = dim_in, out_channels = dim_out, kernel_size = kernel_size),
            nn.SiLU(),
            nn.Conv1d(in_channels = dim_in, out_channels = dim_out, kernel_size = kernel_size)
        )

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )