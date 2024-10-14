import pandas as pd
import numpy as np

df=pd.read_excel("college_data.xlsx")
df.head()

df["Fees"]

#Rename columns for better operations
df.rename(columns={'Computer_ CET_cuttoff ': 'comp_cet_cutoff'}, inplace=True)

df["comp_cet_cutoff"]

df["comp_cet_cutoff"].shape

df.dtypes

#perform imputation to fill NaN values
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
#check the dataframe
df['comp_cet_cutoff']=pd.DataFrame(mean_imputer.fit_transform(df[['comp_cet_cutoff']]))
df['comp_cet_cutoff'].isna().sum()

#convert to float
df.comp_cet_cutoff = df.comp_cet_cutoff.astype(float)
df.dtypes

import matplotlib.pyplot as plt
import seaborn as sns

#check for outliers in required columns
plt.boxplot(df['comp_cet_cutoff'])

plt.boxplot(df['Fees'])

#remove the outliers using IQR method
def detect_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

# Detecting outliers in multiple columns
outliers = detect_outliers_iqr(df[['comp_cet_cutoff','Fees']])

# Remove rows where any column has an outlier
df_cleaned = df[~outliers.any(axis=1)]

#Normalize the data
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm = norm_func(df_cleaned.iloc[:, [2, 5]])

df_norm

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

#identifying the ideal number of clusters using TWSS and elbow curve
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)

    TWSS.append(kmeans.inertia_)#total sum of square

TWSS

plt.plot(k,TWSS,'ro-');
plt.xlabel("No_of_clusters");
plt.xlabel("Total_within_SS")

#Preprocessing using Min Max Scaler
scaler=MinMaxScaler()
scaler.fit(df[['comp_cet_cutoff']])
df['comp_cet_cutoff']=scaler.transform(df[['comp_cet_cutoff']])

scaler.fit(df[['Fees']])
df['Fees']=scaler.transform(df[['Fees']])

plt.scatter(df.Fees,df['comp_cet_cutoff'])
df.head()

plt.scatter(df.Fees,df['comp_cet_cutoff'])

km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Fees','comp_cet_cutoff']])
y_predicted

df['cluster']=y_predicted
df.head()
km.cluster_centers_

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
plt.scatter(df1.Fees,df1['comp_cet_cutoff'],color='green')
plt.scatter(df2.Fees,df2['comp_cet_cutoff'],color='red')
plt.scatter(df3.Fees,df3['comp_cet_cutoff'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',s=200,label='centroid')
plt.xlabel('Fees')
plt.ylabel('comp_cet_cutoff')
plt.legend()
