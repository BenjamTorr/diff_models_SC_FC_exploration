import torch

class corticalSCFCDataset(torch.utils.data.Dataset):
    """
        Data for the diffusion model 
    """

    def __init__(self, FC_matrices, SC_matrices, data_transform=None, condition_transform=None):
        """
            Initialization FC is X and SC is condition 
        """
        assert len(FC_matrices) == len(SC_matrices), "FC and SC must have the same length"
        self.FC_matrices = FC_matrices
        self.SC_matrices = SC_matrices
        self.data_transform = data_transform
        self.condition_transform = condition_transform

    def __len__(self):
        return len(self.FC_matrices)

    def __getitem(self, idx):
        data = self.FC_matrices[idx]
        condition = self.SC_matrices[idx]
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if not isinstance(data, torch.Tensor):
            condition = torch.tensor(condition, dtype=torch.float32)
        
        return data, condition