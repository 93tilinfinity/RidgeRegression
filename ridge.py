import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.model_selection import KFold,train_test_split

# Build Dataset
path = 'dataset_Facebook.csv'
data = pd.read_csv(path,delimiter=';')
data = data.drop(columns=['Paid','Category','Post Month','Post Weekday','Post Hour','Type',\
                       'comment','share','Total Interactions'])
data = data.dropna()
data_std = (data - data.mean())/data.std()
X = data_std.drop(columns = ['like'])
y = data_std['like']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Build Models
RR1 = Ridge(0.01)
RR2 = Ridge(1.0)
RR3 = Ridge(10.0)
RR4 = Ridge(100.0)
RR5 = Ridge(1000.0)
LS = LinearRegression()
models = {'LS':LS,'Ridge0.01':RR1,'Ridge1.0':RR2,'Ridge10.0':RR3,'Ridge100.0':RR4,'Ridge1000.0':RR5}

# 10-fold Cross Validated Error/Std
def build_errors(n_splits=10):
    kf = KFold(n_splits)
    CV_score = []
    CV_coef = []
    for train_index, test_index in kf.split(X_train):
        X_train_CV, X_test_CV = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_CV, y_test_CV = y_train.iloc[train_index], y_train.iloc[test_index]
        for model in models.values():
            model.fit(X_train_CV, y_train_CV)
        CV_score.append([m.score(X_test_CV,y_test_CV) for m in models.values()])
        CV_coef.append([m.coef_ for m in models.values()])

    error = [np.array([s[i] for s in CV_score]).mean() for i in range(len(models))]
    coef_std = [np.array([c[i] for c in CV_coef]).std() for i in range(len(models))]
    return pd.DataFrame({'CV_Rsq':error,'CV_coef_std':coef_std},index=models.keys())

# Fit Models
def fit(models):
    for model in models.values():
        model.fit(X_train, y_train)
    idx = ['Intercept']+X_train.columns.to_list()
    return pd.DataFrame({model[0]:np.insert([model[1].intercept_],[0],model[1].coef_)\
                          for model in models.items()},index=idx)

# Plot coef evolution
def plot_coefficients(df):
    plt.style.use('seaborn-darkgrid')
    for col in df.index.to_list():
        if col != 'Intercept':
            plt.plot(df.T.index,df.loc[col],marker='o',markersize=5,linewidth=1.75,linestyle='dashed')
    plt.legend(loc='lower right',prop={'size': 8})
    plt.xlabel('Regression Type')
    plt.ylabel('Coefficient')
    plt.show()

df_error = build_errors(10)
df = fit(models)
plot_coefficients(df)

# summary all data
full_df = df.T.join(df_error)
# pd.set_option('display.max_columns',20)
# print(full_df.T)
