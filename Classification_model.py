from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_class_frac, plot_sig_frac
import Watchmal_dependencies.binning as bins


def weighted_f1(y_true, y_pred, w=1):
    precision = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    mod_f1 = (1 + w) * (precision * rec) / (w * precision + rec) if (w * precision + rec) != 0 else 0
    return mod_f1, precision, rec


def run_model(df, train_col, train_labels, model_name='gbdt', grid_search=False):
    model = CutEngine(df, train_col, train_labels, model_name, grid_search)
    model.prepare_data()

    plt.hist(model.y_train, label='train_sig', histtype='step')
    plt.hist(model.y_test, label='test_sig', histtype='step')
    plt.xlabel('Signal/background label')
    plt.ylabel('Count')
    plt.title('Signal/background counts used for testing and training')
    plt.legend()
    plt.show()
    plt.clf()

    if grid_search:
        model.grid_search()
    model.train()
    model.make_calibration_curve()

    test_acc, test_precision, test_recall, test_f1 = model.test()
    print(f' The best threshold is : {model.best_thresh}')
    train_acc, train_f1, train_precision, train_recall = model.test_on_train()

    model.plot_probs(label='train', y_log=True)
    print(f'The train accuracy is:{train_acc}')
    print(f'The train f1 is:{train_f1}')
    print(f'The train precision is:{train_precision}')
    print(f'The train recall is:{train_recall}')

    model.plot_probs(label='test', y_log=True)
    print(f'The test accuracy is:{test_acc}')
    print(f'The test f1 is:{test_f1}')
    print(f'The test precision is:{test_precision}')
    print(f'The test recall is:{test_recall}')

    feature_imp = model.get_features_importance()
    for key in feature_imp:
        print(f'{key} : {feature_imp[key]}')

    model.plot_roc(mode='rejection', label=model_name)
    model.plot_roc(mode='efficiency', label=model_name)

    df[model_name + '_sig'] = 0
    df.loc[df['h5_labels'].isin(train_labels), model_name + '_sig'] = (model.y_prob >= model.best_thresh).astype(int)

    df_cut = df[df['h5_labels'] == 1]

    mom_binning = bins.get_binning(df_cut['h5_momentum'], 20, 0, 1000)
    reco_mom_e_binning = bins.get_binning(df_cut['reco_electron_mom'], 20, 0, 1000)
    dwall_binning = bins.get_binning(df_cut['h5_dwall'], 22, 50, 300)
    towall_binning = bins.get_binning(df_cut['h5_towall'], 30, 50, 800)

    model.plot_efficiency_profile(mom_binning, y_label="Signal Efficiency",
                                  x_label="true_mom", legend='best', y_lim=None, errors=True)
    model.plot_efficiency_profile(reco_mom_e_binning, y_label="Signal Efficiency",
                                  x_label="mom_e", legend='best', y_lim=None, errors=True)
    model.plot_efficiency_profile(dwall_binning, y_label="Signal Efficiency",
                                  x_label="dwall", legend='best', y_lim=None, errors=True)
    model.plot_efficiency_profile(towall_binning, y_label="Signal Efficiency",
                                  x_label="towall", legend='best', y_lim=None, errors=True)


class CutEngine:
    def __init__(self, df, train_col, train_labels=None, model_name='gbdt', grid_search=False):

        self.model_name = model_name

        if grid_search:
            self.param_grid_gbdt = {
                'n_estimators': [i for i in np.arange(100, 300, 20)],
                'subsample': [i for i in np.arange(0.1, 0.9, 0.08)],
                'tol': [10 ** -i for i in np.arange(1, 9, 0.8)],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
            }

            self.param_grid_adaboost = {
                'n_estimators': [i for i in np.arange(50, 200, 20)],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
            }

            self.param_grid_xgboost = {
                'n_estimators': [i for i in np.arange(100, 300, 20)],
                'subsample': [i for i in np.arange(0.1, 0.9, 0.08)],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                'max_depth': [3, 4, 5]
            }

            if model_name == 'gbdt':
                self.param_grid = self.param_grid_gbdt
                self.rf_model = GradientBoostingClassifier()

            elif model_name == 'adaboost':
                self.param_grid = self.param_grid_adaboost
                self.rf_model = AdaBoostClassifier()

            elif model_name == 'xgboost':
                self.param_grid = self.param_grid_xgboost
                self.rf_model = XGBClassifier()
        else:
            if model_name == 'gbdt':
                self.rf_model = GradientBoostingClassifier(n_estimators=280, subsample=0.1, tol=2.51*10**-7,
                                                           learning_rate=0.01)

            elif model_name == 'adaboost':
                self.rf_model = AdaBoostClassifier(n_estimators=280, learning_rate=0.01)

            elif model_name == 'xgboost':
                self.rf_model = XGBClassifier(n_estimators=280, subsample=0.1, learning_rate=0.01, max_depth=5)

        print(f"Using {model_name} model")

        if train_labels is None:
            self.training_labels = [1, 2, 3]

        self.model_name = model_name
        self.training_labels = train_labels
        self.best_thresh = 0
        self.training_col = train_col
        self.df = df
        self.features = [key for key in train_col if key != 'true_sig']

        self.X = None
        self.X_scaled = None
        self.y = None
        self.X_train = None
        self.X_train_scaled = None
        self.X_test = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

        self.rf_model_cal = CalibratedClassifierCV(self.rf_model, cv=5, method='isotonic')

        self.y_prob = None
        self.test_prob = None
        self.train_prob = None
        self.test_predictions = None
        self.train_predictions = None

    def prepare_data(self):
        df_cut = self.df[self.df['h5_labels'].isin(self.training_labels)]
        df_chosen = df_cut[self.training_col]

        plot_class_frac(df_cut)
        plot_sig_frac(df_chosen)

        self.X = df_chosen.drop(columns=['true_sig'])
        self.y = df_chosen['true_sig']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.9,
                                                                                random_state=42)

        scaler = StandardScaler()

        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.X_scaled = scaler.transform(self.X)

    def grid_search(self):
        grid_search = GridSearchCV(self.rf_model, self.param_grid, cv=5, error_score=0, refit=True, scoring='accuracy',
                                   verbose=0, n_jobs=-1)
        grid_search.fit(self.X_train_scaled, self.y_train)
        self.rf_model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")

    def train(self):
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        self.rf_model_cal.fit(self.X_train_scaled, self.y_train)

        self.test_prob = self.rf_model.predict_proba(self.X_test_scaled)[:, 1]
        self.train_prob = self.rf_model.predict_proba(self.X_train_scaled)[:, 1]
        self.y_prob = self.rf_model.predict_proba(self.X_scaled)[:, 1]

    def test(self):

        threshold = np.linspace(0, 1, 100)
        predictions = (self.test_prob[:, np.newaxis] >= threshold).astype(int)
        array = np.array([weighted_f1(self.y_test, prediction) for prediction in predictions.T])
        f1_array = array[:, 0]
        precision_array = array[:, 1]
        recall_array = array[:, 2]

        best_threshold_index = np.argmax(f1_array)

        plt.plot(threshold, precision_array, label=self.model_name + ' precision')
        plt.plot(threshold, recall_array, label=self.model_name + ' recall')
        plt.plot(threshold, f1_array, label=self.model_name + ' f1_score')
        plt.xlabel('Threshold')
        plt.title('Precision, recall and f1_score for different thresholds')
        plt.legend()
        plt.show()
        plt.clf()

        self.best_thresh = threshold[best_threshold_index]
        self.test_predictions = (self.test_prob >= self.best_thresh).astype(int)
        test_accuracy = accuracy_score(self.y_test, self.test_predictions)
        return test_accuracy, precision_array[best_threshold_index], recall_array[best_threshold_index], f1_array[best_threshold_index]

    def test_on_train(self):
        self.train_predictions = (self.train_prob >= self.best_thresh).astype(int)
        train_accuracy = accuracy_score(self.y_train, self.train_predictions)
        train_f1, prec, rec = weighted_f1(self.y_train, self.train_predictions)
        return train_accuracy, train_f1, prec, rec

    def plot_probs(self, label='test', y_log=False):
        if label == 'train':
            plt.hist([self.train_prob[self.y_train == i] for i in range(2)], histtype="step", bins=50, range=(0, 1),
                     label=["Background", "Signal"])
        else:
            plt.hist([self.test_prob[self.y_test == i] for i in range(2)], histtype="step", bins=50, range=(0, 1),
                     label=["Background", "Signal"])
        plt.title(f'Probability Distribution for {label} events')
        plt.xlabel("Computed probability")
        plt.ylabel("Number of events")
        if y_log:
            plt.yscale('log')
        plt.legend()
        plt.show()
        plt.clf()

    def make_calibration_curve(self):
        prob_true, prob_pred = calibration_curve(self.y_test, self.rf_model_cal.predict_proba(self.X_test_scaled)[:, 1],
                                                 n_bins=10, strategy='uniform')

        plt.plot(prob_pred, prob_true, marker='o', label='Calibrated Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='black')

        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.show()

    def get_features_importance(self):
        imp = self.rf_model.feature_importances_
        dic_imp = {}
        for i in range(len(self.features)):
            dic_imp[self.features[i]] = imp[i]
        return dic_imp

    def plot_roc(self, x_label="Signal_Efficiency", y_label="Background_Rejection",
                 x_lim=None, y_lim=None, y_log=None, x_log=None, legend='best', mode='rejection', **plot_args):

        fig, ax = plt.subplots(1, 1)
        fpr, tpr, _ = metrics.roc_curve(self.y, self.y_prob)
        auc = metrics.auc(fpr, tpr)
        args = {**plot_args}
        args['label'] = f"{args['label']} (AUC={auc:.4f})"
        if mode == 'rejection':
            if y_log is None:
                y_log = True
            with np.errstate(divide='ignore'):
                ax.plot(tpr, 1 / fpr, **args)
        elif mode == 'efficiency':
            ax.plot(fpr, tpr, **args)
            x_label = "Background_Mis_ID rate"
            y_label = "Signal Efficiency"
        else:
            raise ValueError(f"Unknown ROC curve mode '{mode}'.")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_log:
            ax.set_xscale('log')
        if y_log:
            ax.set_yscale('log')
        if y_lim:
            ax.set_ylim(y_lim)
        if x_lim:
            ax.set_xlim(x_lim)
        if legend:
            ax.legend(loc=legend)
        return fig, ax

    def plot_efficiency_profile(self, binning, x_label="",
                                y_label="", legend='best', y_lim=None, errors=True,  **plot_args):
        df_cut = self.df[self.df['h5_labels'].isin(self.training_labels)]
        fig, ax = plt.subplots()
        plot_args.setdefault('lw', 2)
        binned_values = bins.apply_binning((self.y_prob > self.best_thresh)[df_cut['h5_labels'] == 1], binning)
        x = bins.bin_centres(binning[0])
        if errors:
            y_values, y_errors = bins.binned_efficiencies(binned_values, errors, reverse=False)
            x_errors = bins.bin_halfwidths(binning[0])
            plot_args.setdefault('marker', '')
            plot_args.setdefault('capsize', 4)
            plot_args.setdefault('capthick', 2)
            ax.errorbar(x, y_values, yerr=y_errors, xerr=x_errors, **plot_args)
        else:
            y = bins.binned_efficiencies(binned_values, errors, reverse=False)
            plot_args.setdefault('marker', 'o')
            ax.plot(x, y, **plot_args)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if legend:
            ax.legend(loc=legend)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        return fig, ax
