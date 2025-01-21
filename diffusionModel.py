import torch
from torch.nn import nn

class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, matrix_hw=(68, 68)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.matrix_chw = matrix_hw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1) * eta
        return noisy

    def backward(self, x, t, condition):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t, condition)
    
