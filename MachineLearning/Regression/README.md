# Regression

This is a type of supervised machine learning technique whose aim is predicting results of continuous variables based on a labeled set of data. In other words, *the process of predicting a continuous value*.

- Y: dependent variable (state, target or final goal)
- X: independent/explanatory variables (causes)

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

A scatterplot clearly shows the relation between variables.

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