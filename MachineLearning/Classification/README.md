# Classification

Classification is a supervised learning approach, which can be thought of as a means of categorizing or "classifying" some unknown items into a discrete set of "classes."

Target attribute in classification is a categorical variable with discrete values.
- Binary classification
- Multi-class classification

Classification has different business use cases as well, for example:
- To predict the category to which a customer belongs
- For Churn detection, where we predict whether a customer switches to another provider or brand 
- Or to predict whether or not a customer responds to a particular advertising campaign.

- Examples: email filtering, speech recognition, handwriting
recognition, bio-metric identification, document classification, and much more.

Types of classification algorithms in ML:
- Covered in this course
    - K-Nearest Neighbors (KNN)
    - Decision Trees 
    - Support Vector Machines
    - Logistic Regression
- Others
    - Naïve Bayes
    - Linear Discriminant Analysis
    - Neural Networks

## K-Nearest Neighbors (KNN)

The k-nearest-neighbors algorithm is a classification algorithm that takes a bunch of labelled points and uses them to learn how to label other points.
- This algorithm classifies cases based on their similarity to other cases.
- Data points that are near each other are said to be **neighbors**
- Based on **Similar cases with the same class labels
are near each other** Thus, the **distance between two cases is a measure of their dissimilarity**. (Different ways of calculate this distance)

Algorithm:
1. Pick a value for K.
2. Calculate the distance from the new case (holdout from each of the cases in the dataset).
3. Search for the K observations in the training data that are ‘nearest’ to the measurements
of the unknown data point.
4. predict the response of the unknown data point using the most popular response value from the K nearest neighbors.

### Calculate similarities

We have to normalize our feature set to get the accurate dissimilarity measure.
Types dissimilarity measures are highly dependent on data type and also the domain that classification is done for it.
- **Minkowski distance**: d = square(sum_of_features(x1i - x2i)^2)

### K number

- A low value of K (K=1) causes a highly complex model as well, which might result in over-fitting of the model. For example, it can be chosen the noise or the anomaly in the data.
- A very high value of K, such as K=20, then the model becomes overly generalized.

The general solution is to reserve a part of your data for testing the **accuracy of the model**.
- Repeat this process, increasing the k, and see which k is best for your model.

### Continuous targets

Nearest neighbors analysis can also be used to compute values for a **continuous target**.
In this situation, the average or median target value of the nearest neighbors is used to obtain the predicted value for the new case.

## Model Evaluation Metrics

Evaluation metrics for classifiers which explain the performance of a model. (capability and accuracy to classify)

Evaluation metrics provide a key role in the development of a model, as they provide insight to areas that might require improvement.

Compare actual values (y) with predicted labels. (y^)

Three model evaluation metrics covered here:
- **Jaccard index** / Jaccard similarity coefficient
- **F1-score (confussion matrix)**: This matrix shows the corrected and wrong predictions, in comparison with the actual labels.
  - A good thing about the confusion matrix is that it shows the model’s ability to correctly predict or separate the classes.
  - *Precision* is a measure of the accuracy, provided that a class label has been predicted.
  - *Recall* is the true positive rate.
  - *F1 score* is the harmonic average of the precision and recall.
- **Log Loss** / Logarithmic loss measures the performance of a classifier where the predicted output is a probability value between 0 and 1.

## Decision Trees

The basic intuition behind a decision tree is to map out all possible decision paths in the form of a tree.
Decision trees are built by splitting the training set into distinct nodes, where one node contains all of, or most of, one category of the data.
Decision trees are about testing an attribute and branching the cases, based on the result of the test.
- Each *internal node* corresponds to a test. Ex/ Sex
- And each *branch* corresponds to a result of the test. Ex/ Male or female
- And each *leaf node* assigns a patient to a class.


A decision tree can be constructed by considering the attributes one by one.
1. Choose an attribute from our dataset.
2. Calculate the significance of the attribute in the splitting of the data. (evaluate if it's an effective attribute or not)
3. Split the data based on the value of the best attribute.
4. Go to each branch and repeat it for the rest of the attributes.

### Implementing Decision trees
- Decision trees are built using recursive partitioning to classify the data.
- The algorithm chooses the most predictive feature to split the data on: determine “which attribute is the
best, or more predictive, to split data based on the feature.”
  - More predictiveness
  - Less impurity
  - Lower entropy
- Selection of the attribute to split is all about "**purity**" of the leaves: A node in the tree is considered “pure” if, in 100% of the cases, the nodes fall into a specific category of the target field.
  - In fact, the method uses recursive partitioning to split the training records into segments by minimizing the “impurity” at each step.
- **”Impurity”** of nodes is calculated by **“Entropy”** of data in the node: Entropy is the amount of information disorder, or the amount of randomness in the data.
  - If the samples are completely homogeneous the entropy is zero and if the samples are equally divided, it has an entropy of one.
- **Information gain** is the information that can increase the level of ertainty after splitting.





