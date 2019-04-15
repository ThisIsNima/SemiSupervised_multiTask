import pandas as pd
import pickle
from scipy.stats import fisher_exact
from sklearn.metrics import confusion_matrix


def computer_fish_p_val(feature, response):
    cm = confusion_matrix(response, feature)
    _, p_val = fisher_exact(cm)

    return p_val


data = pickle.load(open('data/2017_data.pkl', 'rb'))

n_row, n_col = data.shape

fisher_p_val = data.iloc[:, :-1].apply(lambda x: computer_fish_p_val(x, data.dropout))

