import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch

def plot_correlation_matrices(real_matrices, predicted_matrices, condition_matrices, labels=None):
    """
    Plots comparison heatmaps for batches of real and predicted correlation matrices.

    Args:
        real_matrices (torch.Tensor or np.ndarray): Tensor of shape [batch, L, L] for real correlation matrices.
        predicted_matrices (torch.Tensor or np.ndarray): Tensor of shape [batch, L, L] for predicted correlation matrices.
        labels (list, optional): Labels for axes. If None, defaults to ["Var1", "Var2", ..., "VarL"].

    Returns:
        None: Displays the Plotly figure.
    """
    
    # Ensure inputs are numpy arrays
    real_matrices = np.array(real_matrices)
    predicted_matrices = np.array(predicted_matrices)
    condition_matrices = np.array(condition_matrices)
    
    # Validate shapes
    if real_matrices.shape != predicted_matrices.shape:
        raise ValueError("real_matrices and predicted_matrices must have the same shape.")
    if len(real_matrices.shape) != 3:
        raise ValueError("Input tensors must have shape [batch, L, L].")

    batch, L, _ = real_matrices.shape

    # Default axis labels
    if labels is None:
        labels = [f"Var{i+1}" for i in range(L)]

    # Create a subplot grid with 2 columns for each matrix pair
    fig = make_subplots(
        rows=batch, cols=3,
        subplot_titles=[f"Real Matrix {i+1}" if j == 1 
                        else f"Predicted Matrix {i+1}" if j == 2 
                        else f"Condition Matrix {i+1}" 
                        for i in range(batch) for j in range(1, 4)],
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )

    # Loop through the batch and add heatmaps
    for i in range(batch):
        # Real matrix heatmap
        if i == 0:
            cb = dict(title="Real" if i == 0 else "", len=0.33)
        else:
            cb = None
        fig.add_trace(
            go.Heatmap(
                z=real_matrices[i],
                x=labels,
                y=labels,
                colorscale="RdBu",
                zmin=-1, zmax=1,
                colorbar=cb
            ),
            row=i + 1, col=1
        )

        # Predicted matrix heatmap
        #scaled = (predicted_matrices[i] - predicted_matrices[i].min()) / predicted_matrices[i].max()
        fig.add_trace(
            go.Heatmap(
                z=predicted_matrices[i],
                x=labels,
                y=labels,
                colorscale="RdBu",
                zmin=-1, zmax=1,
                colorbar=None
            ),
            row=i + 1, col=2
        )

        fig.add_trace(
            go.Heatmap(
                z=condition_matrices[i],
                x=labels,
                y=labels,
                colorscale="RdBu",
                zmin=-1, zmax=1,  # Shared scale for condition matrices
                colorbar=None
            ),
            row=i + 1, col=3
        )

    # Update layout
    fig.update_layout(
        title="Comparison of Real and Predicted Correlation Matrices",
        height=300 * batch,  # Adjust height dynamically based on batch size
        width=1500,
    )

    fig.show()


def plot_noise_process(ddpm, x0, n_plots=10):
    """

    """
    L, _ = x0.shape
    x0 = torch.tensor(x0.reshape(1, 1, L, L)).to(ddpm.device)
    n_steps = ddpm.n_steps
    t = np.linspace(0, n_steps - 1, n_plots)
    # Default axis labels

    labels = [f"Var{i+1}" for i in range(L)]
    
    
    # Create a subplot grid with 2 columns for each matrix pair
    fig = make_subplots(
        rows=1, cols=n_plots,
        subplot_titles=[f"Noisy matrix at t = {t[i]}"  for i in range(n_plots) ],
        horizontal_spacing=0.01,
        vertical_spacing=0.1
    )


    # Loop through the batch and add heatmaps
    for i in range(n_plots):
        # Real matrix heatmap
        #time = torch.tensor([t[i]]).reshape(1,-1).to(ddpm.device)
        noisy = ddpm.forward(x0, int(np.floor(t[i])))[0].reshape(L,L).cpu()
        fig.add_trace(
            go.Heatmap(
                z=noisy,
                x=labels,
                y=labels,
                colorscale="Viridis"
            ),
            row=1, col= i + 1
        )

    # Update layout
    fig.update_layout(
        title="Noisy process for FC",
        height=500 ,  # Adjust height dynamically based on batch size
        width= 500 * n_plots,
    )

    fig.show()