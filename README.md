Machine Learning Projects – Core Algorithms
This repository contains clean, practical implementations of key supervised machine learning algorithms. Each mini-project focuses on a specific ML technique using real-world-like datasets, clear code, and basic model evaluation. Some projects also include pipelines for better preprocessing and reproducibility.

## Projects Overview
1. Linear Regression
Used for predicting continuous values such as salary, price, etc.

Goal: Predict outcomes based on a linear relationship

Concept: y = mx + c (best-fit line using least squares)

Includes: Data preprocessing, evaluation with RMSE

2. Logistic Regression
Binary classification algorithm (Yes/No, 0/1)

Goal: Predict probability of class membership

Concept: Sigmoid function (outputs between 0 and 1)

Includes: Confusion matrix, precision, recall, F1-score

3. Support Vector Machine (SVM)
Used for classification with high-dimensional and margin-based data.

Goal: Find optimal hyperplane to separate classes

Concept: Maximize margin between class boundaries

Includes: Kernel tricks, grid search

4. K-Nearest Neighbors (KNN)
Instance-based learning that uses distance to predict class.

Goal: Classify a point based on its k nearest neighbors

Concept: Euclidean distance, majority voting

Includes: Scaling, hyperparameter tuning (K)

5. Naive Bayes
Probabilistic classifier based on Bayes’ theorem.

Goal: Classify text or categorical data (e.g., spam detection)

Concept: P(class | data) ∝ P(data | class) * P(class)

Includes: Text preprocessing, CountVectorizer

6. Decision Tree
Tree-like structure to split data using logical conditions.

Goal: Build decision rules from features to classify

Concept: Gini Impurity or Entropy to split nodes

Includes: Visualizations, pruning, overfitting check

7. Random Forest
Ensemble of decision trees that improves model accuracy.

Goal: Aggregate predictions from multiple decision trees

Concept: Bagging + Ensemble = More robust predictions

Includes: Feature importance, random state control

## Advanced Additions (Pipeline Projects)
These projects include robust preprocessing pipelines using scikit-learn's Pipeline and ColumnTransformer.

Regression with Pipelines: Imputation, Scaling, Feature Engineering

Classification with Pipelines: Encoding, Scaling, OneHotEncoder

Custom Transformers: Built using BaseEstimator and TransformerMixin

##Tools and Libraries Used
Python 

Scikit-learn

NumPy & Pandas

Matplotlib & Seaborn

Jupyter Notebook

Optional: FunctionTransformer, Pipeline, ColumnTransformer

## Learning Outcomes
Developed a solid understanding of core ML algorithms and their applications

Learned to prepare data, train models, and evaluate results

Applied concepts like feature scaling, train/test split, and cross-validation

Practiced using evaluation metrics such as accuracy, precision, recall, F1-score, RMSE, and confusion matrix

Gained experience with modular, clean, and scalable code using pipelines

