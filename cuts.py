def basic(df):
    basic_cuts = (
            (df['reco_electron_dwall'] > 50)
            & (df['reco_electron_towall'] > 100)
            & (df['reco_electron_mom'] > 100)
            & ~((df['h5_labels'] == 2) & ~(df['h5_momentum'] > 2 * df['h5_dwall']))
    )
    return basic_cuts


def fq_emu_base(df):
    cuts = (
            (df['e/mu_likelihood ratio'] > 119.8 - 0.32 * df['reco_electron_mom'])
            & (df['e/mu_likelihood ratio'] > 45.6 - 0.14 * df['reco_electron_dwall'])
            & (df['e/mu_likelihood ratio'] > 149.1 - 0.69 * df['reco_electron_towall'])
    )
    return cuts


def fq_emu(df):
    cuts = (
            (df['e/mu_likelihood ratio'] > 119.2 - 0.38 * df['reco_electron_mom'])
            & (df['e/mu_likelihood ratio'] > 44.5 - 0.25 * df['reco_electron_dwall'])
            & (df['e/mu_likelihood ratio'] > 148.6 - 0.74 * df['reco_electron_towall'])
    )
    return cuts
    

def fq_epi0_base(df):
    cuts = (
            (df['pi0/e_likelihood ratio'] < 32.2 + 0.42 * df['reco_electron_mom'])
            & (df['pi0/e_likelihood ratio'] < 165 - 0.1 * df['pi0_mass'])
            & (df['pi0/e_likelihood ratio'] < 66.2 + 1.92 * df['reco_electron_dwall'])
            & (df['pi0/e_likelihood ratio'] < 56.5 + 0.95 * df['reco_electron_towall'])
    )
    return cuts


def fq_epi0(df):
    cuts = (
            (df['pi0/e_likelihood ratio'] < 36.1 + 0.81 * df['reco_electron_mom'])
            & (df['pi0/e_likelihood ratio'] < 322 - 1.8 * df['pi0_mass'])
            & (df['pi0/e_likelihood ratio'] < 227.9 + 2.79 * df['reco_electron_dwall'])
            & (df['pi0/e_likelihood ratio'] < 70.9 + 02.79 * df['reco_electron_towall'])
    )
    return cuts
    

def ml_emu_base(df):
    cuts = (
        (df['pmu'] < 10 ** (-0.6))
    )
    return cuts


def ml_epi0_base(df):
    cuts = (
            (df['ppi0'] < 10 ** (-0.6))
            & (df['pe'] > 10 ** (-1.2))
    )
    return cuts
