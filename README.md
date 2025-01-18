# Machine Learning Algorithms on Titanic Dataset
This repository explores the application of various Machine Learning algorithms for classification, regression, and clustering using the Titanic dataset. It includes code implementations for data preprocessing, training, and evaluation of models.

## Table of Contents
- Introduction
- Features and Preprocessing
- Algorithms Implemented
  - Classification Models
  - Regression Models
  - Clustering
- Visualization
- How to Run
- Requirements
- License
## Introduction
The Titanic dataset is used to predict passenger survival (classification) and explore other relationships (e.g., Fare vs. Survival) through regression and clustering models.

## Features and Preprocessing
Feature Used:
  - Sex: Encoded as binary values (0 for female, 1 for male).
  - Fare: Used for regression and clustering.

Handling Missing Data:
  - Numerical features are filled with their mean values.

Data Splitting:
  - Train-test split with 20% test data.

## Algorithms Implemented
### Classification Models
- Logistic Regression: Predicts survival based on the passenger's sex.
- Naive Bayes: Classifies passengers into survived or not based on their sex.
- Decision Tree Classifier: Creates a decision tree for survival prediction.
- Random Forest Classifier: Uses an ensemble of decision trees for classification.

### Regression Models
- Linear Regression: Predicts survival probability based on fare.
- Polynomial Regression: Fits a second-degree polynomial to model Fare vs. Survival.
- Decision Tree Regressor: Non-linear regression for survival prediction.
- Random Forest Regressor: An ensemble method for regression tasks.

### Clustering
- K-Means Clustering: Groups passengers into clusters based on sex and fare.

## Visualization
Classification Visualizations:
  - Logistic Regression plot: Survival probability based on sex.

Regression Visualizations:
  - Linear Regression: Predicted vs. actual survival based on fare.
  - Polynomial Regression: Enhanced regression curve for fare vs. survival.

Clustering Visualization:
  - K-Means: Scatter plot with two clusters and centroids.

## How to Run
1. Install Required Libraries:

````bash
pip install numpy pandas matplotlib scikit-learn
````

2. Run the Script:
Place the Titanic dataset as train.csv in the same directory and run:

````bash
python script_name.py
````
3. View the Output:

- Accuracy and Mean Squared Error metrics will be printed to the console.
- Plots for models and clustering will be displayed.
## Requirements
- Python 3.x
- Libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-learn
## License
This project is licensed under the MIT License. See the LICENSE file for details.
