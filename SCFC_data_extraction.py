import numpy as np
from scipy.io import loadmat


def load_cortical_FC(FC_path = 'data/desikan_fc_all.mat', transform_FC = True):
    #load data from mat format
    data_FC = loadmat(FC_path)
    #we only get the cortical as the only information available for SC
    cortical_vars = data_FC['ChosenROI_cortical'][0]
    # Functional connectivity for the cortical columns/rows
    cortical_FC = np.array([ FC[cortical_vars,:][:, cortical_vars] for FC in data_FC['all_fc'][0] if len(FC) > 1 ])
    # ids for people with non-empty FC
    present_ids = data_FC['subjList'].reshape(1065)
    ids = np.array([ present_ids[i] for i in range(1065) if len(data_FC['all_fc'][0][i]) > 1 ])

    if transform_FC:
        for i in range(cortical_FC.shape[0]):
            matrix = cortical_FC[i]
            np.fill_diagonal(matrix, matrix.diagonal() - 1e-5)
            cortical_FC[i] = np.atanh(matrix)
        


    return ids, cortical_FC

def load_cortical_SC(SC_path = 'data/HCP_cortical_DesikanAtlas.mat', transform_SC = True):
    #load data from mat format
    data_SC = loadmat(SC_path)
    #we get the matrices directly and we also check they are not empty
    cortical_SC = np.array([ data_SC['hcp_sc_count'][:,:,i] for i in range(1065) if len(data_SC['hcp_sc_count'][:,:,i]) > 1])
    # we divide each row by the max value in its row
    if transform_SC:
        max_elements = np.max(cortical_SC, axis = (1,2), keepdims=True)
        cortical_SC = cortical_SC / max_elements
        
    #symmetrize
    cortical_SC = np.array([cortical_SC[idx] + cortical_SC[idx].transpose() for idx in range(len(cortical_SC))])
    #we get the ids
    present_ids = data_SC['all_id'].reshape(1065)
    ids = np.array([ present_ids[i] for i in range(1065) if len(data_SC['hcp_sc_count'][:,:,i]) > 1])

    return ids, cortical_SC

def load_cortical_SC_FC(FC_path = 'data/desikan_fc_all.mat',
                        SC_path = 'data/HCP_cortical_DesikanAtlas.mat',
                        transform_SC = True,
                        transform_FC = True):
    #we first load both information
    ids_FC, cortical_FC = load_cortical_FC(FC_path, transform_FC)
    ids_SC, cortical_SC = load_cortical_SC(SC_path, transform_SC)
    #we match ids FC only as it has less
    cortical_SC_filt = np.array([ cortical_SC[i] for i in range(1065) if ids_SC[i] in ids_FC])
    ids_SC_filt = np.array([ ids_SC[i] for i in range(1065) if ids_SC[i] in ids_FC])
    #check everything is fine
    assert cortical_SC_filt.shape == cortical_FC.shape
    assert np.array_equal(ids_SC_filt, ids_FC)

    return ids_FC, cortical_FC, cortical_SC_filt

