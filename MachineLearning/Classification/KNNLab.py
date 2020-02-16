import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

# When using the 'inline' backend, your matplotlib graphs will be included in your notebook, next to the code.
# Used in Jython
# %matplotlib inline

# Load dataset
df = pd.read_csv('teleCust1000t.csv')
df.head()

# Study a certain value
df['custcat'].value_counts()

# Explore data with an histogram
df.hist(column='income', bins=50)

# See column names
df.columns


# To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
# Define feature sets
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)

# Define dependent variable
y = df['custcat'].values

### Normalize Data
# Data Standardization give data zero mean and unit variance, it is good practice, 
# especially for algorithms such as KNN which is based on distance of cases
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

### Train/Test Split -> evaluate out-of-sample accuracy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

### K nearest neighbor (KNN)
from sklearn.neighbors import KNeighborsClassifier

## Training 
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

## Predicting 
yhat = neigh.predict(X_test)
yhat[0:5]

## Accuracy evaluation
# __accuracy classification score__ is a function that computes subset accuracy.
# This function is equal to the jaccard_similarity_score function.
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat)) # out-of-sample accuracy

### Practice. Evaluate model with K = 6
#Train Model and Predict  
neigh6 = KNeighborsClassifier(n_neighbors = 6).fit(X_train,y_train)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, neigh6.predict(X_test))) # out-of-sample accuracy


### Test for different Ks
Ks = 200
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
# ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

# Plot model accuracy for Different number of Neighbors
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

