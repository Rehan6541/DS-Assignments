import pandas as pd        # For reading and manipulating the dataset.
import numpy as np        # For numerical operations.
import matplotlib.pyplot as plt   #For plotting and visualizing data.
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler   #For normalizing the dataset.
from sklearn.cluster import KMeans     # For applying the K-Means clustering algorithm.

df=pd.read_excel("C:\Kmeans_assign\EastWest\EastWestAirlines.xlsx")
df

# Step 1: Apply K-Means Clustering
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Balance', 'Qual_miles']])

# Step 2: Add the cluster labels to the DataFrame
df['cluster'] = y_predicted
y_predicted
# Step 3: Visualize the clusters
# Separate the data based on cluster labels
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plot the clusters
plt.scatter(df1.Balance, df1['Qual_miles'], color='orange', label='Cluster 0')
plt.scatter(df2.Balance, df2['Qual_miles'], color='blue', label='Cluster 1')
plt.scatter(df3.Balance, df3['Qual_miles'], color='green',label='Cluster 2')

# Plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            color='purple', marker='*', s=200, label='Centroid')

# Set plot title and labels
plt.title('K-Means Clustering')
plt.xlabel('Balance')
plt.ylabel('Qual_miles')
plt.legend()
plt.show()

#  preprocessing

scaler=MinMaxScaler()
scaler.fit(df[['Balance']])
df['Balance']=scaler.transform(df[['Balance']])

scaler.fit(df[['Qual_miles']])
df['Qual_miles']=scaler.transform(df[['Qual_miles']])

df.head()

plt.scatter(df['Balance'],df['Qual_miles'])

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Balance', 'Qual_miles']])

# Step 2: Add the cluster labels to the DataFrame
df['cluster'] = y_predicted
y_predicted
# Step 3: Visualize the clusters
# Separate the data based on cluster labels
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]


# Plot the clusters
plt.scatter(df1.Balance, df1['Qual_miles'], color='orange')#, label='Cluster 0')
plt.scatter(df2.Balance, df2['Qual_miles'], color='blue')#, label='Cluster 1')
plt.scatter(df3.Balance, df3['Qual_miles'], color='green')#,label='Cluster 2')

# Plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            color='purple', marker='*', s=200, label='Centroid')

# Set plot title and labels
plt.title('K-Means Clustering')
plt.xlabel('Balance')
plt.ylabel('Qual_miles')
plt.legend()
plt.show()


##ELBOW CURVE
# Importing required libraries
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_excel("C:\Kmeans_assign\EastWest\EastWestAirlines.xlsx")

# Drop any non-numeric columns, if necessary (e.g., State column)
# In this case, we assume no such columns exist, so we'll skip this step.
# However, if needed, use: df = df.drop(["State"], axis=1)

# Select relevant columns for clustering
columns_to_cluster = ['Balance', 'Qual_miles', 'Bonus_miles', 'Bonus_trans', 
                      'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll']

# Apply normalization to the selected columns
def norm_func(i):
    return (i - i.min()) / (i.max() - i.min())

# Normalize the selected columns
df_norm = norm_func(df[columns_to_cluster])

# Determine the ideal number of clusters using the Elbow Method
TWSS = []
k = list(range(2, 8))  # Cluster range to test

for i in k:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)  # Total within-cluster sum of squares (inertia)

# Plotting the Elbow Curve
plt.plot(k, TWSS, 'ro-')
plt.xlabel("Number of Clusters")
plt.ylabel("Total Within-Cluster Sum of Squares (TWSS)")
plt.title("Elbow Curve for Optimal Number of Clusters")
plt.show()

# Based on the elbow curve, let's assume the ideal number of clusters is 3
model = KMeans(n_clusters=3, random_state=42)
model.fit(df_norm)

# Assign the cluster labels to the original dataframe
df['clust'] = model.labels_

# Reorganize the dataframe to show the cluster label first
df = df[['clust'] + columns_to_cluster]

# Display the first few rows of the clustered dataframe
print(df.head())

# Calculate the mean of each feature within each cluster
cluster_means = df.groupby('clust').mean()
print(cluster_means)

# Save the clustered data to a CSV file
df.to_csv("kmeans_eastwestairlines.csv", encoding="utf-8", index=False)

# Check the current working directory to ensure the file is saved in the right place
import os
print(os.getcwd())


