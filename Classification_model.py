from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_class_frac, plot_sig_frac


def weighted_f1(y_true, y_pred, w=1):
    # gives more importance to recall over precision in F1 score

    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    mod_f1 = (1 + w) * (prec * rec) / (w * prec + rec)
    return mod_f1, prec, rec


def run_model(df, train_col, train_labels, model_name='gbdt', grid_search=False):
    model = CutEngine(df, train_col, train_labels, model_name)
    model.prepare_data()
    if grid_search:
        model.grid_search()
    model.train()

    test_acc, test_f1, test_precision, test_recall = model.test()
    train_acc, train_f1, train_precision, train_recall = model.test_on_train()

    plt.hist(model.y_train, label='train_sig')
    plt.hist(model.y_test, label='test_sig')
    plt.xlabel('Signal/background label')
    plt.ylabel('Count')
    plt.title('Signal/background counts used for testing and training')
    plt.legend()
    plt.show()
    plt.clf()

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

    feature_imp = model.get_features_importances()
    print(feature_imp)

    df['Classified_sig'] = (model.y_prob >= model.best_thresh).astype(int)


class CutEngine:
    def __init__(self, df, train_col, train_labels=None, model_name='gbdt'):

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
            print(f"Using {model_name} model")
            self.rf_model = GradientBoostingClassifier(n_estimators=280, subsample=0.1, tol=2.51*10**-7,
                                                       learning_rate=0.01)

        elif model_name == 'adaboost':
            self.param_grid = self.param_grid_adaboost
            print(f"Using {model_name} model")
            self.rf_model = AdaBoostClassifier(n_estimators=280, learning_rate=0.01)

        elif model_name == 'xgboost':
            self.param_grid = self.param_grid_xgboost
            print(f"Using {model_name} model")
            self.rf_model = XGBClassifier(n_estimators=280, subsample=0.1, tol=2.51*10**-7, learning_rate=0.01,
                                          max_depth=5)

        if train_labels is None:
            self.training_labels = [1, 2, 3]

        self.training_labels = train_labels
        self.best_thresh = 0
        self.training_col = train_col
        self.df = df
        self.features = [key for key in train_col if key != 'sig']

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
        self.predictions = None

    def prepare_data(self):
        df_cut = self.df[self.df['h5_labels'].isin(self.training_labels)]
        df_chosen = df_cut[self.training_col]

        plot_class_frac(df_cut)
        plot_sig_frac(df_chosen)

        self.X = df_chosen.drop(columns=['true_sig'])
        self.y = df_chosen['true_sig']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.8,
                                                                                random_state=42)

        scaler = StandardScaler()

        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.X_scaled = scaler.transform(self.X)

    def grid_search(self):
        grid_search = GridSearchCV(self.rf_model, self.param_grid, cv=5, error_score=0, refit=True, scoring='accuracy',
                                   verbose=2, n_jobs=-1)
        grid_search.fit(self.X_train_scaled, self.y_train)
        self.rf_model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")

    def train(self):
        # Train the chosen classifier
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        self.rf_model_cal.fit(self.X_train_scaled, self.y_train)

        self.test_prob = self.rf_model.predict_proba(self.X_test_scaled)[:, 1]
        self.train_prob = self.rf_model.predict_proba(self.X_train_scaled)[:, 1]
        self.y_prob = self.rf_model.predict_proba(self.X_scaled)[:, 1]

    def test(self, weight=1):
        # Evaluate on the test set
        best_f1 = 0
        for threshold in np.linspace(0, 0.9, 100):
            test_predictions = (self.test_prob >= threshold).astype(int)
            test_f1, prec, rec = weighted_f1(self.y_test, test_predictions, w=weight)
            if test_f1 > best_f1:
                best_f1 = test_f1
                self.best_thresh = threshold
        self.predictions = (self.test_prob >= self.best_thresh).astype(int)
        self.predictions = (self.test_prob >= self.best_thresh).astype(int)
        test_accuracy = accuracy_score(self.y_test, self.predictions)
        test_f1, prec, rec = weighted_f1(self.y_test, self.predictions, w=weight)
        return test_accuracy, test_f1, prec, rec

    def test_on_train(self, weight=1):
        # Evaluate on the train set
        train_predictions = (self.train_prob >= self.best_thresh).astype(int)
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        train_f1, prec, rec = weighted_f1(self.y_train, train_predictions, w=weight)
        return train_accuracy, train_f1, prec, rec

    def plot_probs(self, label='test', y_log=False):
        plt.hist([self.test_prob[self.y_test == i] for i in range(2)], histtype="step", bins=50, range=(0, 1),
                 label=["Background", "Signal"])
        if label == 'train':
            plt.hist([self.train_prob[self.y_train == i] for i in range(2)], histtype="step", bins=50, range=(0, 1),
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

    def get_features_importances(self):
        imp = self.rf_model.feature_importances_
        dic_imp = {}
        for i in range(len(self.features)):
            dic_imp[self.features[i]] = imp[i]
        return dic_imp
