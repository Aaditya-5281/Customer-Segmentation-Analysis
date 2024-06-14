import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing

# Set the styles to Seaborn
sns.set()

# Load the data
data = pd.read_csv('3.12 Example.csv')

# Scatter plot of the two variables
plt.scatter(data['Satisfaction'], data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

# Select both features by creating a copy of the data
x = data.copy()

# K-means clustering
kmeans = KMeans(4)
kmeans.fit(x)

# Create a copy of the input data with predicted clusters
clusters = x.copy()
clusters['cluster_pred'] = kmeans.fit_predict(x)

# Plot the data with clusters
plt.scatter(clusters['Satisfaction'], clusters['Loyalty'], c=clusters['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

# Scale the inputs
x_scaled = preprocessing.scale(x)

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

# Plot the number of clusters vs WCSS
plt.plot(range(1, 10), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# K-means clustering with 9 clusters
kmeans_new = KMeans(9)
kmeans_new.fit(x_scaled)

# Create a new data frame with the predicted clusters
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)

# Plot the new clusters
plt.scatter(clusters_new['Satisfaction'], clusters_new['Loyalty'], c=clusters_new['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
