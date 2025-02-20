import torch
import torch.nn as nn

class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, matrix_chw=(1, 68, 68)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.matrix_chw = matrix_chw
        self.c = matrix_chw[0]
        self.h = matrix_chw[1]
        self.w = matrix_chw[2]
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))], device = self.device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t].to(self.device)

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)
        if eta.device != self.device:
            eta.to(self.device)
            
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t, condition):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t, condition)
    
    def sample(self, condition, n_samples = 16):
        """We will sample a matrix from a condition matrix"""
    
        with torch.no_grad():
            #We start from random noise
            x = torch.randn(n_samples, self.c, self.h, self.w).to(self.device)
            condition = condition.to(self.device)

            for idx, t in enumerate(list(range(self.n_steps))[::-1]):
                time_tensor = (torch.ones(n_samples, 1) * t).to(self.device).long()
                eta_theta = self.backward(x, time_tensor, condition)

                alpha_t = self.alphas[t]
                alpha_t_bar = self.alpha_bars[t]

                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
                if t > 0:
                    z = torch.randn(n_samples, self.c, self.h, self.w).to(self.device)

                    # Option 1: sigma_t squared = beta_t
                    beta_t = self.betas[t]
                    # sigma_t = beta_t.sqrt()

                    # Option 2: sigma_t squared = beta_tilda_t
                    prev_alpha_t_bar = self.alpha_bars[t-1] if t > 0 else self.alphas[0]
                    beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                    sigma_t = beta_tilda_t.sqrt()

                    # Adding some more noise like in Langevin Dynamics fashion
                    x = x + sigma_t * z
            #we transform to -1 to 1
            return x.reshape(n_samples, self.h, self.w)






    
