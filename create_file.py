import numpy as np
import pandas as pd
from utils import get_h5_fq_offsets
import h5py
from Watchmal_dependencies import math, fq_output


def create_true_df(file='/home/pdeperio/machine_learning/data/'
                   'IWCD_mPMT_Short/IWCD_mPMT_Short_emgp0_E0to1000MeV_digihits.h5'):

    idxs_path = ('/home/pdeperio/machine_learning/data/IWCD_mPMT_Short/index_lists/'
                 '4class_e_mu_gamma_pi0/IWCD_mPMT_Short_4_class_3M_emgp0_idxs.npz')

    test_idxs = np.load(idxs_path, allow_pickle=True)['test_idxs']
    h5_file = h5py.File(file, 'r')

    events_hits_index = np.append(h5_file['event_hits_index'], h5_file['hit_pmt'].shape[0])
    h5_nhits = (events_hits_index[test_idxs + 1] - events_hits_index[test_idxs]).squeeze()

    df = pd.DataFrame()
    df['h5_labels'] = np.array(h5_file['labels'])[test_idxs].squeeze()
    df['h5_dwall'] = math.dwall(np.array(h5_file['positions'])[test_idxs].squeeze())
    df['h5_towall'] = math.towall(np.array(h5_file['positions'])[test_idxs].squeeze(),
                                  np.array(h5_file['angles'])[test_idxs])
    df['h5_momentum'] = math.momentum_from_energy(np.array(h5_file['energies'])[test_idxs].squeeze(),
                                                  np.array(h5_file['labels'])[test_idxs].squeeze())
    df['h5_vetos'] = np.array(h5_file['veto'])[test_idxs].squeeze()
    df['h5_nhits'] = h5_nhits
    return df


def create_softmax(file_path='/home/surajrai1900/WatChMaL/outputs/2023-09-19/03-59-17/outputs/'):
    indices = file_path + 'indices.npy'
    softmax = file_path + 'softmax.npy'
    label = file_path + 'labels.npy'
    predictions = file_path + 'predictions.npy'

    df = pd.DataFrame()
    df['softmax_predictions'] = np.load(predictions)

    data_softmax = np.load(softmax)
    df['pgamma'] = data_softmax[:, 0]
    df['pe'] = data_softmax[:, 1]
    df['pmu'] = data_softmax[:, 2]
    df['ppi0'] = data_softmax[:, 3]
    return df


def create_fq_df(file_path='/home/pdeperio/machine_learning/data/IWCD_mPMT_Short/fiTQun/'):
    particle_names = ['gamma', 'e-', 'mu-', 'pi0']
    fq_files = [file_path + f'IWCD_mPMT_Short_{i}_E0to1000MeV_unif-pos-R400-y300cm_4pi-dir.fiTQun.root' for i in
                particle_names]
    fq = fq_output.FiTQunOutput(fq_files)

    offsets = get_h5_fq_offsets()

    reco_e_pos = np.array(fq.electron_position)
    reco_e_angles = math.angles_from_direction(np.array(fq.electron_direction))

    df = pd.DataFrame()
    df['reco_electron_mom'] = np.array(fq.electron_momentum)[offsets]
    df['reco_muon_mom'] = np.array(fq.muon_momentum)[offsets]
    df['reco_pi0_mom'] = np.array(fq.pi0_momentum)[offsets]
    df['reco_electron_dwall'] = math.dwall(reco_e_pos)[offsets]
    df['reco_electron_towall'] = math.towall(reco_e_pos, reco_e_angles)[offsets]
    df['e_likelihood'] = np.array(fq.electron_nll)[offsets]
    df['mu_likelihood'] = np.array(fq.muon_nll)[offsets]
    df['pi0_likelihood'] = np.array(fq.pi0_nll)[offsets]
    df['pi0_mass'] = np.array(fq.pi0_mass)[offsets]
    df['e/mu_likelihood ratio'] = df['mu_likelihood'] - df['e_likelihood']
    df['pi0/e_likelihood ratio'] = df['e_likelihood'] - df['pi0_likelihood']
    return df


def relevant_df(true_variables=None,
                reco_variables=None,
                softmax_variables=None,
                true_sig=1):

    if softmax_variables is None:
        softmax_variables = ['pgamma', 'pe', 'pmu', 'ppi0']

    if reco_variables is None:
        reco_variables = ['e/mu_likelihood ratio', 'pi0/e_likelihood ratio', 'e_likelihood', 'mu_likelihood', 'pi0_likelihood', 'reco_electron_mom',
                          'reco_electron_dwall', 'reco_electron_towall', 'pi0_mass']
      
    if true_variables is None:
        true_variables = ['h5_labels', 'h5_momentum', 'h5_towall', 'h5_dwall']

    if not isinstance(true_sig, (list, np.ndarray)):
        true_sig = [true_sig]

    df_true = create_true_df()
    df_fq = create_fq_df()
    df_softmax = create_softmax()

    df = pd.DataFrame({item: df_softmax[item] for item in softmax_variables})
    df[true_variables] = df_true[true_variables]
    df[reco_variables] = df_fq[reco_variables]
    df['true_sig'] = (df['h5_labels'].isin(true_sig)).astype(int)

    return df
