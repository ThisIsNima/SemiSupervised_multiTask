import pandas as pd
import pickle
import numpy as np


def one_hot_transform(d, include_na=True, is_categorical=False, n_bin=4):
    cols = d.columns

    one_hot_data = list()

    for c in cols:

        series = d[c].copy()

        if not is_categorical:
            cat_series = pd.qcut(series, n_bin, duplicates='drop')
        else:
            cat_series = series

        dummy_series = pd.get_dummies(cat_series, prefix=c, dummy_na=include_na)

        one_hot_data.append(dummy_series)

    one_hot_data = pd.concat(one_hot_data, axis=1, sort=False)

    return one_hot_data

# you should create a 'data' folder, and put the data file 'Output.xlsx' there
tbl = pd.read_excel('data/Output.xlsx')
n_row, n_col = tbl.shape
col_names = tbl.columns

# find the percentage of missing for each column
pct_null = tbl.apply(lambda x: sum(pd.isnull(x))*1.0/n_row)

# set the cutoff value. Features with missing percentage greater than this cutoff will be removed from the data set.
missing_cutoff = 0.98
col_names = col_names[pct_null < missing_cutoff]
tbl = tbl[col_names]

# Remove the unnecessary features
survive_indicator = tbl.X15 == 'N'
tbl = tbl.loc[survive_indicator].reset_index()
# remove the 'X1' column
tbl.drop(columns='X1', inplace=True)

# find unique values in the column
n_unique_per_col = tbl.apply(lambda x: len(pd.unique(x)))

questionnaire_related_start = 'RESPONSE_1_NUM'
questionnaire_related_end = 'RESPONSE_82_NUM'

demographic_related = ['PERCENTILE', 'X9']
education_related = ['College 1 X21', 'Curriculum X26']

financial_related_start = 'X34'
financial_related_end = 'X35'

financial_need_related_start = 'X37'
financial_need_related_end = 'X38'

credit_related = ['Attempted X6']

# extract data from the table
i1 = np.where(col_names == questionnaire_related_start)[0][0]
i2 = np.where(col_names == questionnaire_related_end)[0][0]
data = one_hot_transform(tbl.iloc[:, i1:i2+1], is_categorical=True)

data = pd.concat((data, one_hot_transform(tbl[education_related], is_categorical=True)), axis=1)

# One-hot Encode the numerical values
data = pd.concat((data, one_hot_transform(tbl[school_score_related])), axis=1)

i1 = np.where(col_names == financial_related_start)[0][0]
i2 = np.where(col_names == financial_related_end)[0][0]
data = pd.concat((data, one_hot_transform(tbl.iloc[:, i1: i2+1])), axis=1)

i1 = np.where(col_names == financial_need_related_start)[0][0]
i2 = np.where(col_names == financial_need_related_end)[0][0]
data = pd.concat((data, one_hot_transform(tbl.iloc[:, i1: i2+1])), axis=1)

data = pd.concat((data, one_hot_transform(tbl[credit_related])), axis=1)

data = pd.concat((data, pd.DataFrame(1 - tbl.RG2*tbl.RG3, columns=['dropout'])), axis=1)

with open('data/2017_version_1.pkl', 'wb') as f:
    pickle.dump(data, f)
