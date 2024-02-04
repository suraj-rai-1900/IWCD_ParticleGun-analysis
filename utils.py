import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py


def get_h5_fq_offsets(h5_true_file='/home/pdeperio/machine_learning/data/IWCD_mPMT_Short/'
                      'IWCD_mPMT_Short_emgp0_E0to1000MeV_digihits.h5'):

    idxs_path = ('/home/pdeperio/machine_learning/data/IWCD_mPMT_Short'
                 '/index_lists/4class_e_mu_gamma_pi0/IWCD_mPMT_Short_4_class_3M_emgp0_idxs.npz')

    test_idxs = np.load(idxs_path, allow_pickle=True)['test_idxs']
    h5_file = h5py.File(h5_true_file, 'r')

    h5_root_files = np.array(h5_file['root_files'])[test_idxs].squeeze()
    h5_event_ids = np.array(h5_file['event_ids'])[test_idxs].squeeze()

    root_file_index = dict.fromkeys(h5_root_files)
    root_file_index.update((k, i) for i, k in enumerate(root_file_index))
    root_file_indices = np.vectorize(root_file_index.__getitem__)(h5_root_files)
    h5_fq_offsets = 3000 * root_file_indices + h5_event_ids

    return h5_fq_offsets


def normalize(df, sig_label, bg_label):
    if not isinstance(sig_label, (list, np.ndarray)):
        sig_label = [sig_label]
    if not isinstance(bg_label, (list, np.ndarray)):
        bg_label = [bg_label]
    prob_sig = 0
    prob_bg = 0

    label_map = ['pgamma', 'pe', 'pmu', 'ppi0']
    for label in sig_label:
        prob_sig += df[label_map[label]]
    for label in bg_label:
        prob_bg += df[label_map[label]]

    norm_prob_sig = prob_sig/(prob_sig + prob_bg)
    norm_prob_bg = prob_bg/(prob_sig + prob_bg)
    return norm_prob_sig, norm_prob_bg


def optimum_cut_linear(df, sig_label, bg_label, key1, key2, guess_1=0, guess_2=0, greater_than=True, is_log=False):
    if not isinstance(sig_label, (list, np.ndarray)):
        sig_label = [sig_label]
    if not isinstance(bg_label, (list, np.ndarray)):
        bg_label = [bg_label]

    if is_log:
        a = np.repeat(np.arange(-0.1, 0.1, 0.01), 20)
        b = np.tile(np.arange(-12, 0, 0.6), 20)
    else:
        a = np.arange(guess_1 - 10, guess_1 + 10, 0.01)
        b = np.arange(guess_2 - 100, guess_2 + 100, 0.1)

        # Takes too much memory but gives more accurate cuts
        #     a =  np.repeat(a_, 2000)
        #     b = np.tile(b_, 2000)

    accuracy_array = np.array([])

    cut_values = 10 ** (np.dot(np.array(df[key1]).reshape(-1, 1), a.reshape(1, -1)) + b.reshape(1, -1)) if is_log else \
        np.dot(np.array(df[key1]).reshape(-1, 1), a.reshape(1, -1)) + b.reshape(1, -1)

    cut = (np.array(df[key2]).reshape(-1, 1) > cut_values) if greater_than else \
        (np.array(df[key2]).reshape(-1, 1) < cut_values)

    for row in cut.T:
        df_cut = df[row]
        accuracy = check_accuracy(df_cut, sig_label, bg_label)
        accuracy = 0 if np.isnan(accuracy) else accuracy
        accuracy_array = np.append(accuracy_array, accuracy)

    accuracy_best = max(accuracy_array)
    index = np.argmax(accuracy_array)

    return a[index], b[index], accuracy_best


def optimum_cut_quadratic(df, sig_label, bg_label, key1, key2, guess_1=0, guess_2=0, guess_3=0, greater_than=True,
                          is_log=False):
    if not isinstance(sig_label, (list, np.ndarray)):
        sig_label = [sig_label]
    if not isinstance(bg_label, (list, np.ndarray)):
        bg_label = [bg_label]

    if is_log:
        a = np.tile(np.arange(-12, 0, 0.6), 20)
        b = np.repeat(np.arange(-1, 1, 0.1), 20)
        c = np.repeat(np.arange(-0.1, 0.1, 0.01), 20)
    else:
        a = np.arange(guess_1 - 100, guess_2 + 100, 0.1)
        b = np.arange(guess_2 - 10, guess_1 + 10, 0.01)
        c = np.arange(guess_3 - 1, guess_3 + 1, 0.001)

    accuracy_array = np.array([])

    if greater_than:
        cut_values = np.dot(np.array(df[key1] ** 2).reshape(-1, 1), c.reshape(1, -1)) \
                     + np.dot(np.array(df[key1]).reshape(-1, 1), b.reshape(1, -1)) \
                     + a.reshape(1, -1)
    else:
        cut_values = np.dot(np.array(df[key1] ** 2).reshape(-1, 1), c.reshape(1, -1)) \
                     + np.dot(np.array(df[key1]).reshape(-1, 1), b.reshape(1, -1)) \
                     + a.reshape(1, -1)

    cut = (np.array(df[key2]).reshape(-1, 1) > 10 ** cut_values) if is_log else \
        (np.array(df[key2]).reshape(-1, 1) > cut_values) if greater_than else \
        (np.array(df[key2]).reshape(-1, 1) < 10 ** cut_values)

    for row in cut.T:
        df_cut = df[row]
        accuracy = check_accuracy(df_cut, sig_label, bg_label)
        accuracy = 0 if np.isnan(accuracy) else accuracy
        accuracy_array = np.append(accuracy_array, accuracy)

    accuracy_best = max(accuracy_array)
    index = np.argmax(accuracy_array)

    return a[index], b[index], c[index], accuracy_best


def check_accuracy(df, sig_label, bg_label):
    if not isinstance(sig_label, (list, np.ndarray)):
        sig_label = [sig_label]
    if not isinstance(bg_label, (list, np.ndarray)):
        bg_label = [bg_label]
    sig = np.sum(np.isin(df['h5_labels'], sig_label))
    bg = np.sum(np.isin(df['h5_labels'], bg_label))
    accuracy = sig / ((sig + bg) ** 0.5)
    return accuracy


def plot_sg_bg(df, sig_label, bg_label, key1, key2, x_range, y_range, bin_x, bin_y, logbin_y=False):
    if not isinstance(sig_label, (list, np.ndarray)):
        sig_label = [sig_label]
    if not isinstance(bg_label, (list, np.ndarray)):
        bg_label = [bg_label]

    df_sg = df[(np.isin(df['h5_labels'], sig_label))]
    df_bg = df[(np.isin(df['h5_labels'], bg_label))]

    if logbin_y:
        bins_y = [10 ** x for x in np.linspace(-12, 0, 50)]
        yscale = "log"
    else:
        bins_y = bin_y
        yscale = "linear"

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    ax = axes[0]
    hist, x_edges, y_edges, im = ax.hist2d(df_sg[key1], df_sg[key2], bins=[bin_x, bins_y], range=[x_range, y_range],
                                           norm=mpl.colors.LogNorm())
    ax.set_xlabel(key1, fontsize=15)
    ax.set_ylabel(key2, fontsize=20)
    ax.set_yscale(yscale)
    ax.set_title('Signal', fontsize=20)
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    hist, x_edges, y_edges, im = ax.hist2d(df_bg[key1], df_bg[key2], bins=[bin_x, bins_y], range=[x_range, y_range],
                                           norm=mpl.colors.LogNorm())
    ax.set_xlabel(key1, fontsize=15)
    ax.set_ylabel(key2, fontsize=20)
    ax.set_yscale(yscale)
    ax.set_title('Background', fontsize=20)
    plt.colorbar(im, ax=ax)

    return fig, axes


def return_basic_cuts(df):
    basic_cuts = (
            (df['reco_electron_dwall'] > 50)
            & (df['reco_electron_towall'] > 100)
            & (df['reco_electron_mom'] > 100)
            & ~((df['h5_labels'] == 2) & ~(df['h5_momentum'] > 2 * df['h5_dwall']))
    )
    return basic_cuts


def return_fq_emu_cuts(df):
    cuts = (
            (df['e/mu_likelihood ratio'] > 119.8 - 0.32 * df['reco_electron_mom'])
            & (df['e/mu_likelihood ratio'] > 45.6 - 0.14 * df['reco_electron_dwall'])
            & (df['e/mu_likelihood ratio'] > 149.1 - 0.69 * df['reco_electron_towall'])
    )
    return cuts


def return_fq_epi0_cuts(df):
    cuts = (
            (df['pi0/e_likelihood ratio'] < 32.2 + 0.42 * df['reco_electron_mom'])
            & (df['pi0/e_likelihood ratio'] < 165 - 0.1 * df['pi0_mass'])
            & (df['pi0/e_likelihood ratio'] < 66.2 + 1.92 * df['reco_electron_dwall'])
            & (df['pi0/e_likelihood ratio'] < 56.5 + 0.95 * df['reco_electron_towall'])
    )
    return cuts


def return_ml_emu_cuts(df):
    cuts = (
        (df['pmu'] < 10 ** (-0.6))
    )
    return cuts


def return_ml_epi0_cuts(df):
    cuts = (
            (df['ppi0'] < 10 ** (-0.6))
            & (df['pe'] > 10 ** (-1.2))
    )
    return cuts


def sg_eff(df, cuts, sg_label):
    if not isinstance(sg_label, (list, np.ndarray)):
        sg_label = [sg_label]
    df_cut = df[cuts]
    total_sig_before_cut = np.sum(np.isin(df['h5_labels'], sg_label))
    selected_as_signal = np.sum(np.isin(df_cut['h5_labels'], sg_label))
    sig_eff = selected_as_signal / total_sig_before_cut
    return sig_eff


def bg_rej(df, cuts, bg_label):
    if not isinstance(bg_label, (list, np.ndarray)):
        bg_label = [bg_label]
    df_cut = df[cuts]
    total_bg_before_cut = np.sum(np.isin(df['h5_labels'], bg_label))
    selected_as_signal = np.sum(np.isin(df_cut['h5_labels'], bg_label))
    bg_re = 1 - selected_as_signal / total_bg_before_cut
    return bg_re


def f1(df, cuts, sig_label, bg_label):
    if not isinstance(sig_label, (list, np.ndarray)):
        sig_label = [sig_label]
    if not isinstance(bg_label, (list, np.ndarray)):
        bg_label = [bg_label]
    cut_int = cuts.astype(int)

    # Calculate precision, recall, and F1-score
    true_positives = np.sum((np.isin(df['h5_labels'], sig_label)) & (cut_int == 1))
    false_positives = np.sum((np.isin(df['h5_labels'], bg_label)) & (cut_int == 1))
    false_negatives = np.sum((np.isin(df['h5_labels'], sig_label)) & (cut_int == 0))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def plot_sel_comp(df, sg_label, bg_label, cut1, cut2, cut1_label, cut2_label):
    label_map = ['gamma', 'e', 'mu', 'pi0']
    color_map = ['cyan', 'red', 'blue', 'green']
    
    if not isinstance(sg_label, (list, np.ndarray)):
        sg_label = [sg_label]
    if not isinstance(bg_label, (list, np.ndarray)):
        bg_label = [bg_label]

    df_cut1 = df[cut1]
    df_cut2 = df[cut2]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    ax = axes[0]
    for label in sg_label:
        ax.hist(df_cut1[df_cut1['h5_labels'] == label]['reco_electron_mom'], bins=15, range=(0, 1500),
                label='sig ' + label_map[label] + cut1_label, histtype='step', linewidth=1.5, color=color_map[label], linestyle='-')
        ax.hist(df_cut2[df_cut2['h5_labels'] == label]['reco_electron_mom'], bins=15, range=(0, 1500),
                label='sig ' + label_map[label] + ' ' + cut2_label, histtype='step', linewidth=1.5, color=color_map[label], linestyle='--')
    for label in bg_label:
        ax.hist(df_cut1[df_cut1['h5_labels'] == label]['reco_electron_mom'], bins=15, range=(0, 1500),
                label='bg ' + label_map[label] + cut1_label, histtype='step', linewidth=1.5, color=color_map[label], linestyle='-')
        ax.hist(df_cut2[df_cut2['h5_labels'] == label]['reco_electron_mom'], bins=15, range=(0, 1500),
                label='bg ' + label_map[label] + ' ' + cut2_label, histtype='step', linewidth=1.5, color=color_map[label], linestyle='--')
    ax.set_xlabel('Reco Momentum', fontsize=20)
    ax.set_ylabel('Event count', fontsize=20)
    ax.set_yscale('log')
    ax.legend()

    ax = axes[1]
    for label in sg_label:
        hist_cut1_sg, sig_x_edges = np.histogram(df_cut1[df_cut1['h5_labels'] == label]['reco_electron_mom'],
                                                 range=(0, 1500), bins=10)
        hist_cut2_sg, sig_x_edges = np.histogram(df_cut2[df_cut2['h5_labels'] == label]['reco_electron_mom'],
                                                 range=(0, 1500), bins=10)
        vals_x_sig = np.array([])
        vals_y_sig = np.array([])
        for j in range(10):
            vals_x_sig = np.concatenate((vals_x_sig, np.linspace(sig_x_edges[j], sig_x_edges[j + 1], 10)))
            vals_y_sig = np.concatenate((vals_y_sig, np.array([hist_cut2_sg[j] / hist_cut1_sg[j] for i in range(10)])))
        ax.plot(vals_x_sig, vals_y_sig, label='Signal ' + label_map[label], linewidth=1.5, linestyle=":", color=color_map[label])

    for label in bg_label:
        hist_cut1_bg, bg_x_edges = np.histogram(df_cut1[df_cut1['h5_labels'] == label]['reco_electron_mom'],
                                                range=(0, 1500), bins=10)
        hist_cut2_bg, bg_x_edges = np.histogram(df_cut2[df_cut2['h5_labels'] == label]['reco_electron_mom'],
                                                range=(0, 1500), bins=10)
        vals_x_bg = np.array([])
        vals_y_bg = np.array([])
        for j in range(10):
            vals_x_bg = np.concatenate((vals_x_bg, np.linspace(bg_x_edges[j], bg_x_edges[j + 1], 10)))
            vals_y_bg = np.concatenate((vals_y_bg, np.array([hist_cut2_bg[j] / hist_cut1_bg[j] for i in range(10)])))
        ax.plot(vals_x_bg, vals_y_bg, label='Background ' + label_map[label], linewidth=1.5, linestyle=":", color=color_map[label])

    ax.set_xlabel("Reco Momentum", fontsize=20)
    ax.set_ylabel(f'{cut2_label}/{cut1_label}', fontsize=20)
    ax.hlines(1, 0, 1500, colors='black', linewidth=2)
    ax.set_ylim(0, 3)
    ax.legend()

    plt.show()

    return fig, axes
  

def plot_class_frac(df):
    label_list = [0, 1, 2, 3]
    legends = ['gamma', 'e', 'mu', 'pi0']
    counts = [len(df[df["h5_labels"] == label]) / len(df) for label in label_list]
    plt.bar(legends, counts)
    plt.xlabel("Particle Type")
    plt.ylabel("Class Fraction [a.u.]")
    plt.title("Class Fraction Distribution")
    plt.xticks(rotation=30, ha='right')
    plt.show()
    plt.clf()


def plot_sig_frac(df):
    label_list = [0, 1]
    legends = ['Background', 'Signal']
    counts = [len(df[df['true_sig'] == label]) / len(df) for label in label_list]
    plt.bar(legends, counts)
    plt.xlabel("Class")
    plt.ylabel("Class Fraction [a.u.]")
    plt.title("Signal and Background Fraction Distribution")
    plt.xticks(rotation=30, ha='right')
    plt.show()
    plt.clf()
