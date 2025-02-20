import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, clear_output



def training_loop(ddpm, loader, n_epochs, optim, device, store_path="models/ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps
    best_epoch = 0

    # Prepare for dynamic plot
    losses = []

    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss over Epochs')


    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            x0, condition = batch
            x0 = x0.to(device)
            condition = condition.to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy matrix based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0.to(device), t.to(device), eta.to(device))

            # Getting model estimation of noise based on the images and the time-step
            #eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1), condition.to(device))
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1), condition.to(device))

            # Optimizing the MSE between the noise plugged and the predicted noise
            #loss = mse(eta_theta, eta)
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)
    
        # Storing the model
        if best_loss > epoch_loss:
            best_epoch = epoch + 1
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
        best_loss_message = f"Best loss at epoch {best_epoch} with loss {best_loss:.4f}"
        

        # Clear previous plot and legend
        ax.clear()
        losses.append(epoch_loss)
        # Clear output and update plot for Jupyter
        clear_output(wait=True)

        max_y = np.percentile(losses, 90)
        min_y = np.min(losses)
        
        # Update plot with new data
        ax.plot(losses, label='Loss', color='blue')
        ax.legend(loc="upper right")
        ax.set_ylim(bottom=min_y, top=max_y)
        
        # Display the updated plot
        display(fig)

        print(best_loss_message)

    plt.show()