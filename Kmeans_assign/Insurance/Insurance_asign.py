##Kmeans clustering

# Importing required libraries
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Kmeans_assign/Insurance/Insurance Dataset.csv")
df.head()
df.columns
# Select relevant columns for clustering
columns_to_cluster = ['Premiums_Paid', 'Age', 'Days_to_Renew', 'Claims_made', 'Income']

# Initial scatter plot (e.g., Age vs Income for quick visualization)
plt.scatter(df['Age'], df['Income'])
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

# Preprocessing using Min-Max Scaler
scaler = MinMaxScaler()

# Fit and transform the selected columns
df_scaled = scaler.fit_transform(df[columns_to_cluster])

# Convert the scaled data back to a DataFrame for easier handling
df_scaled = pd.DataFrame(df_scaled, columns=columns_to_cluster)

# Initialize KMeans
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df_scaled)

# Add the cluster labels to the original dataframe
df['cluster'] = y_predicted

# Display the first few rows of the updated dataframe
print(df.head())

# Display cluster centers in the scaled space
print("Cluster Centers (scaled):")
print(km.cluster_centers_)

# Inverse transform the cluster centers back to the original scale
centroids_original = scaler.inverse_transform(km.cluster_centers_)
print("Cluster Centers (original scale):")
print(centroids_original)

# Creating dataframes for each cluster
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plotting each cluster with different colors (Age vs Income as an example)
plt.scatter(df1['Age'], df1['Income'], color='green', label='Cluster 1')
plt.scatter(df2['Age'], df2['Income'], color='red', label='Cluster 2')
plt.scatter(df3['Age'], df3['Income'], color='black', label='Cluster 3')

# Plotting the cluster centers
plt.scatter(centroids_original[:, 1], centroids_original[:, 4], color='purple', marker='*', s=200, label='Centroid')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()


##ELBOW CURVE
# Importing required libraries
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Kmeans_assign/Insurance/Insurance Dataset.csv")

# Drop any non-numeric columns, if necessary 
# In this case, we assume no such columns exist, so we'll skip this step.
# However, if needed, use: df = df.drop(["____"], axis=1)

# Select relevant columns for clustering
columns_to_cluster = ['Premiums_Paid', 'Age', 'Days_to_Renew', 'Claims_made', 'Income']


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
df.to_csv("kmeans_Insurance.csv", encoding="utf-8", index=False)

# Check the current working directory to ensure the file is saved in the right place
import os
print(os.getcwd())



