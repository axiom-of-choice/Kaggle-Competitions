import matplotlib.pyplot as plt
import  pandas as pd
#%%
# Hola, este script es para analizar el dataset movielens
#%%
# Import the necessary modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%
filepath = "../datasets/ml-latest-small/"
links = pd.read_csv(filepath + "links.csv")
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")
tags = pd.read_csv(filepath + "tags.csv")
#%%
print("Links")
print(links.head())
print("------------------")
print("movies")
print(movies.head())
print("------------------")
print("ratings")
print(ratings.head())
print("------------------")
print("tags")
print(tags.head())
#%%
ratings.drop('timestamp',axis=1,inplace=True)
tags.drop('timestamp',axis=1,inplace=True)
#%%
sns.countplot('rating',data=ratings)
plt.show()
#%%
plt.close()
print(ratings['userId'].value_counts())
plt.show()
#%%
