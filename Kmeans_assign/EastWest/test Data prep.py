#Created on Tue Aug 20 09:06:30 2024

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

1.You are given a dataset with missing values, duplicate entries, and 
inconsistent formatting. Explain the steps you would take to clean this 
dataset and implement the cleaning process using Python. Include 
handling of missing data, removing duplicates.

df=pd.read_csv("C:/Data science/diamonds_messy.csv")
df.head()
df.shape
df.columns
df.dtypes

#df.carat=df.carat.astype(int)
#df.price=df.price.astype(int)

duplicate=df.duplicated("carat")
df.duplicated("carat").sum()
df=df.drop_duplicates("carat")
df.shape

from sklearn.impute import SimpleImputer
mode_imp=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
imp=pd.DataFrame(mode_imp.fit_transform(df[["carat"]]))
Nan=imp.isna()
sum(Nan)

mode_imp=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
imp=pd.DataFrame(mode_imp.fit_transform(df[["carat"]]))
Nan=imp.isna()
sum(Nan)

mode_imp=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
imp=pd.DataFrame(mode_imp.fit_transform(df[["cut"]]))
Nan=imp.isna()
sum(Nan)



4.You have a dataset, identify outliers in the numerical features using 
methods such as Z-score. Discuss the impact of outliers on data analysis 
and machine learning models, and show how to handle these outliers by 
removing them.

sns.boxplot(df.carat)#no outliers
sns.boxplot(df.price)#no outliers
sns.boxplot(df.x)#outliers
sns.boxplot(df.y)#18 cloumn 
sns.boxplot(df.z)#outliers
IQR=df.x.quantile(0.75)-df.x.quantile(0.25)
IQR
lower_limit=df.x.quantile(0.25)-IQR
lower_limit
upper_limit=df.x.quantile(0.75)+IQR
upper_limit

#removing outliers with trimming
import numpy as np
outliers=np.where(df.x>upper_limit,True,np.where(df.x<lower_limit,True,False))
df_trimmed=df.loc[~outliers]
df.shape
df_trimmed.shape
sns.boxplot(df_trimmed.x)

3.Perform the next step of data preprocessing: Encoding the categorical 
variables.





