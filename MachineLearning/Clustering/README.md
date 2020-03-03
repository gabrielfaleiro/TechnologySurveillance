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
- Density-based clustering
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
  3. Assign each object to the closest cluster respect to its centroid selecting an appropiate distance measure
    - Therefore, you will form a matrix where each row represents the distance of a custome from each centroid. It is called the "distance-matrix."
    - Model error can be calculated respect to each centroid: SSE
  4. each cluster center will be updated to be the *mean for data points in its cluster*
  5. Iterate until centroids no longer move

However, as it is a **heuristic algorithm**, **there is no guarantee that it will converge to the global optimum**
To solve this problem, it is common to **run the whole process, multiple times, with different starting conditions**.

Notes:
- K-Means Clustering isn't directly applicable to categorical variables because Euclidean distance function isn't really meaningful for discrete variables

### Evaluate resulting clusters (Accuracy)
External approach:
- Compare results with ground truth (usually not available since it is an unsupervised algorithm)
Internal approach:
- Average distance between data points within a cluster or compared to their centroids

Ussual approach to choose the best k value is to compare metric of accuracy results with their associated k values.
Examples:
- mean distance between data points and their cluster centroid
- however, the larger K the better accuracy
- **Elbow method**: plotting these values will be seen a change in the slope or elbow point, where the rate of decrease sharply shifts. This is the right k value for clustering


## Hierarchical Clustering
Hierarchical clustering algorithms build a hierarchy of clusters where each node is a cluster consisting of the clusters of its daughter nodes.
Strategies for hierarchical clustering generally fall into two types: 
- Divisive: Divisive is top-down, so you start with all observations in a large cluster and break it down into smaller pieces.
- Agglomerative: is bottom-up, where each observation starts in its own cluster and pairs of clusters are merged together as they move up the hierarchy.
The Agglomerative approach is more popular among data scientists.

### Agglomerative Clustering

feature can be multi-dimensional, and distance measurement can be either Euclidean, Pearson, average
distance, or many others, depending on data type and domain knowledge.

New distant measurement is calculated for the merged cluster and the rest of the nodes / cluster

Hierarchical clustering is **typically visualized as a dendrogram**

Characteristics:
- Hierarchical clustering does not require a pre-specified number of clusters.
- However, in some applications we want a partition of disjoint clusters just as in flat clustering, cutting in a specific level of similarity.

1. Create n clusters, one for each data point
2. Compute de Distance / Proximity Matrix (n x n)
3. Repeat
  1. Merge the two nearest clusters
  2. Update the proximity matrix
4. Until only a single cluster remains or the specified number of clusters is reached

the key operation is the computation of the proximity between the clusters with one point, and also clusters with multiple data points.

Distance between points:
- Euclidean distance

Distance between clusters: 
- **Single-Linkage Clustering**: Minimum distance between clusters, between 2 different points of each cluster
- **Complete-Linkage Clustering**: Maximum distance between clusters, between 2 different points of each cluster
- **Average-Linkage Clustering**: Average distance between clusters, average distance of each point from one cluster to every point in another cluster
- **Centroid-Linkage Clustering**: Minimum distance between cluster centroids. Centroids are the average of the feature sets of points in a cluster

In general, it completely depends on the data type, dimensionality of data, and most importantly, the domain knowledge of the dataset.

In fact, different approaches to defining the distance between clusters, distinguish the different algorithms.

Advantages:
1. Does not require a number of lusters to be specified
2. Easy to implement
3. Dendrograms are very useful in understanding the data

Disadvantages:
1. Can never undo any previous steps
2. Generally has long computation times, compared for example with k-Means
3. With large datasets it can be difficult to determine the correct number of clusters by the dendrogram

### Comparing k-Means and Hierarchical clustering

k-Means
- More efficient for large datasets
- Requires the number of clusters to be specified
- Gives only one partitioning of the data based on the predefined number of clusters
- Potentially returns different clusters each time due to random initialisation of centroids

Hierarchical clustering
- Can be slow for large datasets
- Does not require the number of clusters to be specified
- Gives more than one partitioning depending on the resolution, it only has to be divided at any chosen level in the dendrogram
- Always generates the same clusters

## Density-based clustering (DBSCAN)

- with arbitrary shape clusters, or clusters within clusters, traditional techniques (k-means, hierarchical, and fuzzy clustering) might not be able to achieve good results
- no notion of outliers
- locates regions of high density that are separated from one another by regions of low density
  - Density, in this context, is defined as the number of points within a specified radius

*DBSCAN - DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.* 
Advantages:
- DBSCAN, a density-based clustering algorithm, which is appropriate to use when examining spatial data.
- DBSCAN algorithm is that it can find out any arbitrary shape cluster without getting affected by noise
  - It can even find a cluster completely surrounded by a different cluster.
- DBSCAN has a notion of noise, and is robust to outliers.
- does not require one to specify the number of clusters

It works based on 2 parameters: 
- **Radius**: R determines a specified radius that, if it includes enough points within it, we call it a "dense area." 
- **Minimum Points**: M determines the minimum number of data points we want in a neighborhood to define a cluster.


Each point in our dataset can be either a 
- **core** point: if, within R-neighborhood of the point, there are at least M points. 
- **border** point: if:
  a. Its neighborhood contains less than M data points, or
  b. It is reachable from some core point (it is within R-distance
from a core point)
- **outlier** point: any other point, it is not a core point, and also, is not close enough to be reachable from a core point.


The next step is to connect core points that are neighbors, and put them in the same cluster.
So, a cluster is formed as at least one core point, plus all reachable core points, plus all their borders.

