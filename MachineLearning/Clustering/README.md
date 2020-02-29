# Clustering

Clustering means finding clusters in a dataset, unsupervised.
A cluster is group of data points or objects in a dataset that are similar to other objects in the group, and dissimilar to data points in other clusters.

- Classification: labeled data and supervised (preclassified data). Modelling -> Prediction
- Clustering: unlabeled data and unsupervised (pattern searching). Modelling -> Segmentation

General applications:
- exploratory data analysis
- summary generation or reducing the scale
- outlier detection (fraud detection or noise removal)
- pre-processing step for any other further operation of the data

Applications:
- Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar characteristics.
- Fraud detection
- Insurance risk of customers
- Recommending books and movies to new customers
- auto-categorize news based on its content
- Recommend simmilar new articles
- Characterize patient behaviour
- group genes with similar expression patterns
- cluster genetic markers to identify family ties

Clustering algorithms:
- Partitioned-based clustering
  - clustering algorithms that produces sphere-like clusters, such as *k-Means*, *k-Median*, or *Fuzzy c-Means*.
  - relatively efficient: used for medium to large size databases
- Hierarchical clustering
  - produce trees of clusters, such as *Agglomerative* and *Divisive* algorithms
  - Intuitive and used for small sets of data
- Density based clustering
  - produce arbitrary shaped clusters
  - especially good when dealing with *spatial clusters* or when there is *noise in your dataset*, for example, the *DBSCAN* algorithm.

## K-Means Clustering

- k-Means is a type of partitioning clustering
- It divides the data into k non-overlapping subsets (or clusters) without any cluster-internal structure, or labels. This means, it’s an unsupervised algorithm.
- Objects within a cluster are very similar
- objects across different clusters are very different or dissimilar

it can be shown that instead of a similarity metric, we can use dissimilarity metrics.
In other words, conventionally, the distance of samples from each other is used to shape the clusters.
- **k-Means tries to minimize the “intra-cluster” distances and maximize the “inter-cluster” distances.**
- Of course, we have to normalize our feature set to get the accurate dissimilarity measure.


There are other **dissimilarity measures** 
- it is highly *dependent on data type* 
- and also *the domain that clustering is done for it*.
For example, you may use 
- Euclidean distance
- cosine similarity
- average distance
Indeed, the **similarity measure** highly controls *how the clusters are formed*, so it is *recommended to understand the domain knowledge of your dataset, and data type of features, and then choose the meaningful distance measurement*.

How it works:
- In the first step, we should determine the number of clusters **k**
  - The key concept of the k-Means algorithm is that it randomly picks a center point (centroids of clusters) for each cluster. 
  - *determining* the number of clusters in a data set, or *k, is a hard problem in k-Means* that we will discuss later.
- Choose centroids:
  1. We can randomly choose 3 observations out of the dataset and use these observations as the initial means.
  2. Random points

- Assign each object to the closest cluster respect to its centroid selecting an appropiate distance measure
- Therefore, you will form a matrix where each row represents the distance of a custome from each centroid. It is called the "distance-matrix."
- Model error can be calculated respect to each centroid: SSE

  3. each cluster center will be updated to be the mean for data points in its cluster

However, as it is a **heuristic algorithm**, **there is no guarantee that it will converge to the global optimum**
To solve this problem, it is common to **run the whole process, multiple times, with different starting conditions**.