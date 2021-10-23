#%% import modules
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
#%%
impute_smart(test)
#%%







