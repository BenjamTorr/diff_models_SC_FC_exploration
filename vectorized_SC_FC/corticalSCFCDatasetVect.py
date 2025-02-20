import torch

class corticalSCFCDataset(torch.utils.data.Dataset):
    """
        Data for the diffusion model 
    """

    def __init__(self, FC_matrices, SC_matrices, vectorize = False):
        """
            Initialization FC is X and SC is condition 
        """
        assert len(FC_matrices) == len(SC_matrices), "FC and SC must have the same length"
        self.FC_matrices = torch.tensor(FC_matrices).to(torch.float32)
        self.SC_matrices = torch.tensor(SC_matrices).to(torch.float32)
        self.vectorize =  vectorize

        #we only use the upper triangular part of SC as feature vector
        rows, cols = torch.triu_indices(self.SC_matrices.size(1), self.SC_matrices.size(2), offset=1)

        rows = rows.to(torch.long)
        cols = cols.to(torch.long)

        self.SC_vectors = self.SC_matrices[:, rows, cols]
        self.FC_vectors = self.FC_matrices[:, rows, cols]
        #reshape vector
        SC_shape = self.SC_vectors.size
        self.SC_vectors = self.SC_vectors.reshape(SC_shape(0), 1, SC_shape(1)) # (B, C, L)
        self.FC_vectors = self.FC_vectors.reshape(SC_shape(0), 1, SC_shape(1))
        #reshape matrices
        FC_shape = FC_matrices.shape
        self.SC_matrices = self.SC_matrices.reshape(FC_shape[0], 1 , FC_shape[1], FC_shape[2]) 
        self.FC_matrices = torch.tensor(FC_matrices.reshape(FC_shape[0], 1, FC_shape[1], FC_shape[2])).to(torch.float32) #(B, C, H, W)


    def __len__(self):
        return len(self.FC_matrices)

    def __getitem__(self, idx):
        data = self.FC_matrices[idx]
        if self.vectorize:
            data = self.FC_vectors[idx]
            condition = self.SC_vectors[idx]
        else:
            condition = self.SC_matrices[idx]

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if not isinstance(condition, torch.Tensor):
            condition = torch.tensor(condition, dtype=torch.float32)
        
        return data, condition