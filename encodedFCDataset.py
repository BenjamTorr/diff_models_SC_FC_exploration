import torch

class encodedFCDataset(torch.utils.data.Dataset):
    """
        Data for the diffusion model 
    """

    def __init__(self, FC_encoded, SC_vectors):
        """
            Initialization FC is X and SC is condition 
        """
        assert len(FC_encoded) == len(SC_vectors), "FC and SC must have the same number of instances"
        self.z = FC_encoded
        self.c = SC_vectors

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        data = self.z[idx]
        condition = self.c[idx]
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if not isinstance(condition, torch.Tensor):
            condition = torch.tensor(condition, dtype=torch.float32)
        
        return data, condition