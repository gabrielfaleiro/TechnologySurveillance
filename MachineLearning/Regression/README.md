# Regression

This is a type of supervised machine learning technique whose aim is predicting results of continuous variables based on a labeled set of data. In other words, *the process of predicting a continuous value*.

- Y: dependent variable (state, target or final goal)
- X: independent/explanatory variables (causes) or predictors

**y** **must be a continuous**, non discrete, variable.

**x** can be measured on either a categorical or continuous measurement scale.

## Simple Regression
When one independent variable is used to estimate a dependent variable.
- Simple **Linear** Regression
- Simple **Non-Linear** Regression
## Multiple Regression
When more than one independent variable is present.
- Multiple **Linear** Regression
- Multiple **Non-Linear** Regression

## Regression algorithms

- Ordinal regression
- Poisson regression
- Fast forest quantile regression
- Linear, Polynomial, Lasso, Stepwise, Ridge regression
- Bayesian linear regression
- Neural network regression
- Decision forest regression
- Boosted decision tree regression
- KNN (K-nearest neighbors)

## Linear vs Non-Linear

- Visual inspection: A scatterplot clearly shows the relation between variables. Best representing bivariable plots of each independent variable and dependent variable.
- Correlation coefficient: **> 0.7** for all independent variable, the **linear regression is a good approximation**

A Fit line is traditionally shown as a polynomial.
- **y^ (hat)** response variable
- **x_i** predictors
- **parameters** to adjust or coefficients. The bias coefficient is the independent coefficient of the equation.
- **Error** = y - y^ (residual error)
- Errors
    - Mean absolute error
    - **Mean Square Error (MSE)** / residual sum of squares: mean value of all squared errors. Minimising this ecuation should generated the best fitting parameters for the fit line.
        - Mathematical approach
        - Optimisation approach
    - Root Mean Squared Error (RMSE)
    - R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. Best possible score is 1.0

### Linear
- Fast to calculate
- No parameters tuning
- Highly interpretable

## Multiple linear regression

Multiple independent variables (X) to estimate a dependent variable (y).

- **Possible problems**
  - Identify variables effectiveness on prediction
  - Predicting impacts of changes
- y is a linear combination of independent variables
- Mathematically: y^_hat = parameters/weight vector_T * feature set vector
  - feature set vector: first element is 1 because represents the bias parameter
  - With multiple predictors, the resulting line is called a plane or hyper-plane
- **Methods to estimate the parameters vector**.
  - *Ordinary Least Squares (OLS)*: minimize the MSE using the data as a matrix using linear algebra. A drawback is that with large sets of data it may take a significant amount of time to complete the calculation. Rule of thumb, less than 10k rows in the dataset.
  - *Optimization algorithm*
    - *Gradient Descent*
    - *Stochastic Gradient Descent*
    - *Newton’s Method*
    - Other valid approaches...
- **How many independent variables** to use?
  - Basically, adding too many independent variables without any theoretical justification may result in an over-fit model.
- **Should the independent variable be continuous?**
  - Basically, categorical independent variables can be incorporated into a regression model by converting them into numerical variables.

## Model Evaluation in Regression Model

- **train and test on the same dataset**: test set is a portion of the train-set, high “training accuracy” and a low “out-of-sample accuracy”
- training accuracy: is the percentage of correct predictions that the model makes when using the test dataset.
  - isn't necessarily a good thing
  - result of over-fitting
- out-of-sample accuracy: is the percentage of correct predictions that the model makes on data that the model has NOT been trained on.
- **train/test split**: more accurate evaluation of “out-of-sample accuracy”, highly dependent on which sets of data the model is trained and tested, ensure that you train your model with the testing set afterwards
- **K-Fold Cross-validation**: resolves most of these issues.
  - Subdivide the dataset into 4 or 5 sets or folds. Evaluate the accuracy of each set of data with the train/test split evaluation and calculate the average of all evaluations.

## Regression evaluation metrics

Evaluation metrics are used to explain the performance of a model:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Relative Absolute Error (RAE) or Residual sum of square
- Relative Squared Error (RSE)
- R-squared: not an error, represents how close the data values are to the fitted regression line.

Comparison of:
- Actual values (y)
- Predicted values (y^)

## Non-Linear Regression

Non-linear regression is a method to model a non-linear relationship between the dependent variable and a set of independent variables.
- y ̂ must be a non-linear function of the parameters θ, not necessarily the features x. Ex\ y^ = θ1 + θ2^2 * x
- Least Squares method cannot be used to fit the data to a non-linear model

Options for non-linear relationships:
- Polynomial regression: a polynomial regression model can still be expressed as multiple linear regression. Ex\ x1 = x, x2 = x^2, x3 = x^3 ...
- Non-linear regression
- Transform the dataset



