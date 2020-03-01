import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
# %matplotlib inline

## k-Means on a randomly generated dataset
np.random.seed(0)

X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
# Input
#     n_samples: The total number of points equally divided among clusters.
#     Value will be: 5000
#     centers: The number of centers to generate, or the fixed center locations.
#     Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
#     cluster_std: The standard deviation of the clusters.
#     Value will be: 0.9

# Output
#     X: Array of shape [n_samples, n_features]. (Feature Matrix)
#     The generated samples.
#     y: Array of shape [n_samples]. (Response Vector)
#     The integer labels for cluster membership of each sample.

# Display
plt.scatter(X[:, 0], X[:, 1], marker='.')

## Setting up K-Means
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
# init: Initialization method of the centroids.
# Value will be: "k-means++"
# k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
# n_clusters: The number of clusters to form as well as the number of centroids to generate.
# Value will be: 4 (since we have 4 centers)
# n_init: Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
# Value will be: 12

# fit the KMeans model with the feature matrix we created above
k_means.fit(X)

# Get labels for each point in the model 
k_means_labels = k_means.labels_
k_means_labels

# Get cluster centers
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

## Creating the Visual Plot
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data poitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()


## Internal Practice
# Show accuracy in identifying each dataset to each cluster

# f1_score
from sklearn.metrics import f1_score
f1_score(y, k_means_labels, average='weighted') 

# jaccard index
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y, k_means_labels)

# IMP NOTE: k_means_labels do not fit with y labels

## Practice
# Try to cluster the above dataset into 3 clusters.
# Notice: do not generate data again, use the same dataset as above.
k_means3 = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means3.fit(X)

k_means3_labels = k_means.labels_
k_means3_labels

k_means3_cluster_centers = k_means.cluster_centers_
k_means3_cluster_centers

fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
plt.show()


#### Customer Segmentation with K-Means
## Load data
import pandas as pd
cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()

## Pre-processing
# k-means algorithm isn't directly applicable to categorical variables because Euclidean distance function isn't really meaningful for discrete variables
df = cust_df.drop('Address', axis=1)
df.head()

# Normalizing over the standard deviation
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

## Modeling
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

## Insights
# We assign the labels to each row in dataframe.
df["Clus_km"] = labels
df.head(5)

# We can easily check the centroid values by averaging the features in each cluster.
df.groupby('Clus_km').mean()

# distribution of customers based on their age and income
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
