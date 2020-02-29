import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
# %matplotlib inline 
import matplotlib.pyplot as plt

#Click here and press Shift+Enter
!wget -O ChurnData.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv

## Load Data
churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()

## Data pre-processing and selection
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int') # skitlearn algorithm needs target data to be integer
churn_df.head()

# size of data set
churn_df.shape
# columns of data set
churn_df.columns

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])


## Normalise the dataset
# It is required only when features have different ranges.
# The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values.
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

## Train/Test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

## Modeling (Logistic Regression with Scikit-learn)
# The version of Logistic Regression in Scikit-learn, support regularization.
# Regularization is a technique used to solve the overfitting problem in machine learning models.
# In a nutshell, it makes coefficients more representative in the cost function 
# in the optimisation algorithm than any outlier value or noise avoiding overfitting to 
# non-general values
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# C parameter indicates inverse of regularization strength
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
# REF: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# Predict test set
yhat = LR.predict(X_test)

# Calculate estimates for all classes
yhat_prob = LR.predict_proba(X_test)

## Evaluation
#---- Jaccard index: the size of the intersection divided by the size of the union of two label sets
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

#---- Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

print (classification_report(y_test, yhat))

# Precision is a measure of the accuracy provided that a class label has been predicted. It is defined by: precision = TP / (TP + FP)
# Recall is true positive rate. It is defined as: Recall = TP / (TP + FN)
# F1 score: Now we are in the position to calculate the F1 scores for each label based on the precision and recall of that label.

#---- Log loss
# Log loss( Logarithmic loss) measures the performance of a classifier where the predicted output is a probability value between 0 and 1.
from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)



## Practice
# Try to build Logistic Regression model again for the same dataset, but this time, use different __solver__ and __regularization__ values? What is new __logLoss__ value?

LR = LogisticRegression(C=1, solver='sag').fit(X_train,y_train)

p_yhat = LR.predict(X_test)
p_yhat_prob = LR.predict_proba(X_test)

log_loss(y_test, p_yhat_prob)
