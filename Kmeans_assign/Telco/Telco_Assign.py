"""
Created on Wed Aug 21 19:37:49 2024
"""
# Importing required libraries
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_excel("C:\Kmeans_assign\Telco\Telco_customer_churn.xlsx")
df.head()
df.columns
df.shape

# Select relevant columns for clustering
columns_to_cluster =  [
    'Tenure_in_Months', 
    'Avg_Monthly_Long_Distance_Charges',
    'Avg_Monthly_GB_Download',
    'Monthly_Charge', 
    'Total_Charges', 
    'Total_Extra_Data_Charges',
    'Total_Long_Distance_Charges',
    'Total_Revenue'
]

# Initial scatter plot (e.g., Age vs Income for quick visualization)
plt.scatter(df['Total_Charges'], df['Total_Revenue'])
plt.xlabel('Total_Charges')
plt.ylabel('Total_Revenue')
plt.show()

# Preprocessing using Min-Max Scaler
scaler = MinMaxScaler()

# Fit and transform the selected columns
df_scaled = scaler.fit_transform(df[columns_to_cluster])

# Convert the scaled data back to a DataFrame for easier handling
df_scaled = pd.DataFrame(df_scaled, columns=columns_to_cluster)

# Initialize KMeans
km = KMeans(n_clusters=4, random_state=42)
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
df4 = df[df.cluster == 3]


# Plotting each cluster with different colors (Age vs Income as an example)
plt.scatter(df1['Total_Charges'], df1['Total_Revenue'], color='green', label='Cluster 1')
plt.scatter(df2['Total_Charges'], df2['Total_Revenue'], color='red', label='Cluster 2')
plt.scatter(df3['Total_Charges'], df3['Total_Revenue'], color='black', label='Cluster 3')
plt.scatter(df4['Total_Charges'], df4['Total_Revenue'], color='blue', label='Cluster 4')


# Plotting the cluster centers
plt.scatter(centroids_original[:, 4], centroids_original[:, 7], color='purple', marker='*', s=200, label='Centroid')
plt.xlabel('Total_Charges')
plt.ylabel('Total_Revenue')
plt.legend()
plt.show()


##ELBOW CURVE
# Importing required libraries
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_excel("C:\Kmeans_assign\Telco\Telco_customer_churn.xlsx")

# Drop any non-numeric columns, if necessary 
# In this case, we assume no such columns exist, so we'll skip this step.
# However, if needed, use: df = df.drop(["____"], axis=1)

# Select relevant columns for clustering
columns_to_cluster =  [
    'Tenure_in_Months', 
    'Avg_Monthly_Long_Distance_Charges',
    'Avg_Monthly_GB_Download',
    'Monthly_Charge', 
    'Total_Charges', 
    'Total_Extra_Data_Charges',
    'Total_Long_Distance_Charges',
    'Total_Revenue'
]

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


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

# Load the Telco Customer Churn dataset
telco_data=pd.read_excel("C:\Kmeans_assign\Telco\Telco_customer_churn.xlsx")


# Display the initial data structure
print("Initial DataFrame:")
print(telco_data.head())

# Step 1: Data Preprocessing
# Drop Customer ID as it is not relevant for clustering
telco_data = telco_data.drop('CustomerID', axis=1)

# Identify categorical and numerical columns
categorical_cols = telco_data.select_dtypes(include=['object']).columns
numerical_cols = telco_data.select_dtypes(include=['int64', 'float64']).columns

# Handle missing values (filling with mode for categorical and median for numerical)
telco_data[categorical_cols] = telco_data[categorical_cols].fillna(telco_data[categorical_cols].mode().iloc[0])
telco_data[numerical_cols] = telco_data[numerical_cols].fillna(telco_data[numerical_cols].median())

# Step 2: Encode categorical variables and scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)])

# Apply the transformations
telco_data_transformed = preprocessor.fit_transform(telco_data)

# Convert the transformed data back into a DataFrame for easy manipulation
telco_data_transformed = pd.DataFrame(telco_data_transformed)

print("Transformed DataFrame:")
print(telco_data_transformed.head())

# Step 3: Applying K-Means Clustering
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(telco_data_transformed)
telco_data_transformed['cluster'] = y_predicted

# Visualize the first two principal components
plt.scatter(telco_data_transformed.iloc[:, 0], telco_data_transformed.iloc[:, 1], c=telco_data_transformed['cluster'], cmap='viridis', marker='o')
plt.title('Customer Segments based on Clustering (First two principal components)')
plt.xlabel('Feature 1 (PCA Component)')
plt.ylabel('Feature 2 (PCA Component)')
plt.colorbar(label='Cluster')
plt.show()

# Display cluster centers
print("Cluster Centers:")
print(km.cluster_centers_)

# Optional: Visualize clusters in 2D space (using first two features)
df1 = telco_data_transformed[telco_data_transformed['cluster'] == 0]
df2 = telco_data_transformed[telco_data_transformed['cluster'] == 1]
df3 = telco_data_transformed[telco_data_transformed['cluster'] == 2]

plt.scatter(df1.iloc[:, 0], df1.iloc[:, 1], color='green', label='Cluster 1')
plt.scatter(df2.iloc[:, 0], df2.iloc[:, 1], color='red', label='Cluster 2')
plt.scatter(df3.iloc[:, 0], df3.iloc[:, 1], color='black', label='Cluster 3')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='Centroids')
plt.xlabel('Feature 1 (PCA Component)')
plt.ylabel('Feature 2 (PCA Component)')
plt.title('Clusters Visualization')
plt.legend()
plt.show()


