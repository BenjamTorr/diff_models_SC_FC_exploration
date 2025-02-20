import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from VAEmodel import vae_loss
import torch.nn.functional as F

def vectorize_uppdiag_matrix(X):
    X = torch.tensor(X).to(torch.float32)
    rows, cols = torch.triu_indices(X.size(1), X.size(2), offset=1)
    X_vect = X[:, rows, cols]
    return X_vect.reshape(X_vect.size(0), 1, X_vect.size(1))

def generate_corr_matrix(vector, n=68):
    matrices = torch.zeros((vector.shape[0], n, n))
    row_idx, col_idx = torch.triu_indices(n, n, offset=1)
    for i in range(vector.shape[0]):
        matrices[i][row_idx, col_idx] = vector[i]
        matrices[i][col_idx, row_idx] = vector[i]
        matrices[i].fill_diagonal_(1)
    return matrices


def training_loop_vae(vae, loader, loader_val, n_epochs, optim, device, store_path="models/vae.pt"):
    best_loss = float("inf")
    best_epoch = 0
    
    # Prepare for dynamic plot
    losses = []
    val_losses = []

    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss over Epochs')


    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            vae.train()
            x0, condition = batch
            x0 = x0.to(device)
            condition = condition.to(device)
            reconstructed, mu, logvar = vae(x0)
            reconstructed = reconstructed.to(device)
            mu = mu.to(device)
            logvar = logvar.to(device)

            # Optimizing the MSE between the noise plugged and the predicted noise
            #loss = mse(eta_theta, eta)
            loss = vae_loss(reconstructed.to(device), x0.to(device), mu.to(device), logvar.to(device), device = device)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)
        
        ## validation
        vae.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_val, leave=False, desc=f"Valodation Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
                # Loading data
                x0, condition = batch
                x0 = x0.to(device)
                condition = condition.to(device)
                reconstructed, mu, logvar = vae(x0)
                reconstructed = reconstructed.to(device)
                mu = mu.to(device)
                logvar = logvar.to(device)

                # Optimizing the MSE between the noise plugged and the predicted noise
                #loss = mse(eta_theta, eta)
                val_loss = vae_loss(reconstructed.to(device), x0.to(device), mu.to(device), logvar.to(device), device = device)
                epoch_val_loss += val_loss.item() * len(x0) / len(loader_val.dataset)
        val_losses.append(epoch_val_loss)

        # Storing the model
        if best_loss > epoch_val_loss:
            best_epoch = epoch + 1
            best_loss = epoch_val_loss
            torch.save(vae.state_dict(), store_path)
        best_loss_message = f"Best loss at epoch {best_epoch} with loss {best_loss:.4f}"
        

        # Clear previous plot and legend
        ax.clear()
        losses.append(epoch_loss)
        # Clear output and update plot for Jupyter
        clear_output(wait=True)


        max_y = np.percentile(np.concatenate([losses, val_losses]), 90)
        min_y = np.min(np.concatenate([losses, val_losses]))
        # Update plot with new data
        ax.plot(losses, label='Loss', color='blue')
        ax.plot(val_losses, label="Validation Loss", color = "red")
        ax.legend(loc="upper right")
        ax.set_ylim(bottom=min_y, top=max_y)

        # Display the updated plot
        display(fig)

        print(best_loss_message)

    plt.show()


def training_loop_ae(AutoEncoder, loader, loader_val, n_epochs, optim, device, store_path="models/AutoEncoder.pt"):
    best_loss = float("inf")
    best_epoch = 0
    
    # Prepare for dynamic plot
    losses = []
    val_losses = []

    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss over Epochs')


    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            AutoEncoder.train()
            x0, condition = batch
            x0 = x0.to(device)
            condition = condition.to(device)
            reconstructed = AutoEncoder(x0)
            reconstructed = reconstructed.to(device)

            loss = F.mse_loss(reconstructed, x0).to(device)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)
        
        ## validation
        AutoEncoder.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_val, leave=False, desc=f"Valodation Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
                # Loading data
                x0, condition = batch
                x0 = x0.to(device)
                condition = condition.to(device)
                reconstructed = AutoEncoder(x0).to(device)
                
                val_loss = F.mse_loss(reconstructed, x0).to(device)
                epoch_val_loss += val_loss.item() * len(x0) / len(loader_val.dataset)
        val_losses.append(epoch_val_loss)

        # Storing the model
        if best_loss > epoch_val_loss:
            best_epoch = epoch + 1
            best_loss = epoch_val_loss
            torch.save(AutoEncoder.state_dict(), store_path)
        best_loss_message = f"Best loss at epoch {best_epoch} with loss {best_loss:.4f}"
        

        # Clear previous plot and legend
        ax.clear()
        losses.append(epoch_loss)
        # Clear output and update plot for Jupyter
        clear_output(wait=True)

        # Update plot with new data
        max_y = np.percentile(np.concatenate([losses, val_losses]), 90)
        min_y = np.min(np.concatenate([losses, val_losses]))
        ax.plot(losses, label='Loss', color='blue')
        ax.plot(val_losses, label="Validation Loss", color = "red")
        ax.legend(loc="upper right")
        ax.set_ylim(bottom=min_y, top=max_y)
        # Display the updated plot
        display(fig)

        print(best_loss_message)

    plt.show()