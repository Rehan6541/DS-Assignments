"""
Created on Wed Aug 14 23:27:55 2024
"""

###CLUSTERING
import pandas as pd
import matplotlib.pyplot as plt
crime1=pd.read_csv("crime_data - Copy.csv")
a=crime1.describe()
#We have one column 'State' which really not useful we will drop it
crime=crime1.drop(["City"],axis=1)
'''We know that there is scale difference among the columns,
which we have to remove
whenever there is mixed data apply normalization'''
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
'''now apply this normalization functionto crime datframe 
for all the rows and columns from i until end
since 0 th column has crimeersity name hence skipped'''
df_norm=norm_func(crime.iloc[:,1:])
'''You can check the df_norm dataframe which is scaled
between values of 0 to 1
you can apply describe function to new dataframe'''
b=df_norm.describe()
'''Before you apply clustering you need to plot dendrogram first
Now to create dendrogram we need to measure distance
we have to import linkage'''
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchical or aglomerative clustering
#ref the help for linkage
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8))
plt.title("Hireraarchical clustering dendogram")
plt.xlabel("Index")
plt.ylabel("Distance")
#ref help of dendrogram
#sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#dendrogram()
#Applying aglomerative clustering choosing 5 as clusters from dendrogram
#whatever has been displayed is dendrogram is not clustering
#it is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',metric="euclidean").fit(df_norm)
#apply labels to the clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#Assign this series to the crime dataframe as column and name the column
crime['clust']=cluster_labels
#we want to relocate the column 7 to 0th position
crime1=crime.iloc[:,[4,0,1,2,3]]
#now check the crime dataframe
crime1.iloc[:,2:].groupby(crime.clust).mean()
#from the output cluster 2 has got higest top10
#lowest accept  ratio,best faculty ratio and highest expenses
#highest graduate ratio
crime1.to_csv("crime_agglo_clustering.csv",encoding="utf-8")
import os
os.getcwd

##################################################################
####Kmeans clustering-Centroid
# Importing required libraries
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Loading the crime data
df = pd.read_csv('crime_data.csv')

# Dropping the 'City' column
df=df.drop(columns=df.columns[0])

# Displaying the first few rows of the dataset
df.head()

# Preprocessing using Min-Max Scaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back to a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Apply KMeans clustering
#km = KMeans(n_clusters=3)
#y_predicted = km.fit_predict(df_scaled)
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df_scaled)

# Adding cluster labels to the original dataframe
df['cluster'] = y_predicted

# Displaying the first few rows of the updated dataframe with clusters
df.head()

# Inverse transform the cluster centers back to the original scale
centroids_original = scaler.inverse_transform(km.cluster_centers_)

# Get the coordinates of the cluster centers
km.cluster_centers_

# Creating dataframes for each cluster
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plotting each cluster with different colors
plt.scatter(df1['Murder'], df1['Assault'], color='green', label='Cluster 1')
plt.scatter(df2['Murder'], df2['Assault'], color='red', label='Cluster 2')
plt.scatter(df3['Murder'], df3['Assault'], color='black', label='Cluster 3')

# Plotting the cluster centers
#plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='Centroid')
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], color='purple', marker='*', s=200, label='Centroid')


# Labeling the axes and showing the legend
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.legend()
plt.show()


###################################################################
####KMEANS clusterings-elbow curve
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
crime1=pd.read_csv("crime_data.csv")
crime1.head
crime1.describe()
crime1.columns
#col="City"
#crime1.rename(columns={'Unnamed: 0':col},inplace=True)
crime=crime1.drop(columns=crime1.columns[0])

#you know that there is scale diffrence among the columns ,which we 
#either by  using mormalization or standardizartion

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#now apply this normalization  function to univ datafram,e for all thye 

df_norm=norm_func(crime.iloc[:,1:])
'''
what will be ideal cluster number, will it be 1,2 or 3
'''
TWSS=[]
k=list(range(2,10))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    
    TWSS.append(kmeans.inertia_)#total sum of square
'''
Kmeans inertia also kowns as sum of square Errors  
(or SSE). calculates the sum of distances of all points within a cluster  from centroid of 
of the point . It is the difference betweeen the obserrved value and predicted value


'''
TWSS

plt.plot(k,TWSS,'ro-');
plt.xlabel("No_of_clusters");
plt.xlabel("Total_within_SS")
'''
How to select value of the elbow curve 
when k changes from  2 to 3, then decrease
in twss is higher than 
when k changes from 3 to 4 
when k values changes from 5 to 6 decrease 
'''

model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
crime['clust']=mb
crime.head()
crime=crime.iloc[:,[4,0,1,2,3]]
crime
crime.iloc[:,2:8].groupby(crime.clust).mean()

crime.to_csv("kmeans_crime.csv",encoding="utf-8")
import os
os.getcwd




############ KMEANS CLUSTERRING ############

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("C:/Kmeans_assign/crime/crime_data.csv")
df

# Step 1: Apply K-Means Clustering
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Murder', 'Assault']])

# Step 2: Add the cluster labels to the DataFrame
df['cluster'] = y_predicted
y_predicted
km.cluster_centers_

# Step 3: Visualize the clusters
# Separate the data based on cluster labels
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plot the clusters
plt.scatter(df1.Murder, df1['Assault'], color='orange')#, label='Cluster 0')
plt.scatter(df2.Murder, df2['Assault'], color='blue')#, label='Cluster 1')
plt.scatter(df3.Murder, df3['Assault'], color='green')#,label='Cluster 2')

# Plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            color='purple', marker='*', s=200, label='Centroid')

# Set plot title and labels
plt.title('K-Means Clustering')
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.legend()
plt.show()

#  preprocessing

scaler=MinMaxScaler()
scaler.fit(df[['Murder']])
df['Murder']=scaler.transform(df[['Murder']])

scaler.fit(df[['Assault']])
df['Assault']=scaler.transform(df[['Assault']])

df.head()

plt.scatter(df.Murder,df['Assault'])

# Step 1: Apply K-Means Clustering
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Murder', 'Assault']])

# Step 2: Add the cluster labels to the DataFrame
df['cluster'] = y_predicted
y_predicted
# Step 3: Visualize the clusters
# Separate the data based on cluster labels
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]


# Plot the clusters
plt.scatter(df1.Murder, df1['Assault'], color='orange')
plt.scatter(df2.Murder, df2['Assault'], color='blue')
plt.scatter(df3.Murder, df3['Assault'], color='green')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            color='purple', marker='*', s=200, label='Centroid')

# Set plot title and labels
#plt.title('K-Means Clustering')
plt.title('K-Means Clustering')
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.legend()
plt.show()

































 






