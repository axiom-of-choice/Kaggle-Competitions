#%% import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%
path = "../datasets/house_prices/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
#%%
print(train.head())
train.columns
#%%
print(test.head())
test.columns
#%%
print(train.info())
#%%
print(test.info())
#%% Impute values depending category
def impute_smart(df):
    ''':argument: Dataframe
        :return: Dataframe with no NanN values
    '''
    nulls = train.isnull().sum();
    nulls = nulls[nulls > 0]
    for i in nulls.index:
        if df[i].dtype != 'object':
            df[i].replace(np.nan,df[i].mean(),inplace=True)
        else:
            df[i].replace(np.nan, df[i].value_counts().index[0],inplace=True)
    return "Imputed done"
#%%
impute_smart(train)
impute_smart(test)
#%%
train.describe()
#%%Dummy variables
train_dummies = pd.get_dummies(train)
test_dummies = pd.get_dummies(test)
#%%Split features and variable
X = train_dummies.drop('SalePrice',axis=1)
y = train.SalePrice
#%%Scale data
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(X)
X_norm = normalizer.transform(X)
#%% Split the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_norm,y,test_size=0.7,random_state=2021)
#%%Import Models
from sklearn.linear_model import LinearRegression
lr = LinearRegression(positive=True)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
#%%Import scores
from sklearn.metrics import mean_squared_error as MSE
mse = MSE(y_test,y_pred)**(1/2)
print("The MSE for the first model is {}".format(mse))
#%%
from matplotlib import  pyplot as plt
plt.hist(y_pred,alpha=0.5,bins=50)
plt.hist(y_test,alpha=0.5,bins=50)
plt.legend(['y_pred','y_test'])
plt.show()










