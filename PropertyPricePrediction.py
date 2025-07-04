'''
Part A: Property Price Prediction

1. Overview
This project focuses on predicting property prices in various districts of California using
several district-level features. By building a predictive model, we aim to identify key variables
that influence housing prices and improve the accuracy of house value predictions. The
project will specifically utilize simple linear regression and multiple linear regression to
address this regression task, ensuring proper data handling and evaluation of the models.

2. Problem Statement
The objective is to predict the median house value in California districts based on features
such as income, the number of rooms, geographical location, and proximity to the ocean.
Given the dataset, we will develop regression models, evaluate their performance, and
determine which model provides the best balance between predictive accuracy and
interpretability                        '''

# Property Price Prediction - Corrected Pipeline

# 1. Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 2. Load Data
df = pd.read_csv('C:\\Users\\Arnav Singla\\Downloads\\land_data.csv.csv') 
print("First 5 rows of dataset:")
print(df.head())

# 3. Basic Info
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# 4. Drop missing values
df.dropna(inplace=True)

# 5. Encode categorical variable if it exists
if 'ocean_proximity' in df.columns:
    le = LabelEncoder()
    df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])

# 6. Check correlation
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 7. Visualize top features against target
target = 'median_house_value'
features_to_plot = ['median_income', 'total_rooms', 'housing_median_age']

for feature in features_to_plot:
    if feature in df.columns:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[feature], y=df[target])
        plt.title(f'{feature} vs {target}')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.show()

# 8. Prepare Features and Target
X = df.drop(columns=[target])
y = df[target]

# Ensure all X columns are numeric
assert X.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all(), "Non-numeric data found in X"

# 9. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Simple Linear Regression (only 'median_income')
if 'median_income' in X_train.columns:
    X_train_simple = X_train[['median_income']]
    X_test_simple = X_test[['median_income']]

    lr_simple = LinearRegression()
    lr_simple.fit(X_train_simple, y_train)

    y_pred_simple = lr_simple.predict(X_test_simple)

    print("\n--- Simple Linear Regression ---")
    print("MSE:", mean_squared_error(y_test, y_pred_simple))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_simple)))
    print("R² Score:", r2_score(y_test, y_pred_simple))

    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred_simple, alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Simple Linear Regression: Actual vs Predicted")
    plt.show()
else:
    print("Feature 'median_income' not found for simple regression.")

# 11. Multiple Linear Regression (using all features)
lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)
y_pred_multi = lr_multi.predict(X_test)

print("\n--- Multiple Linear Regression ---")
print("MSE:", mean_squared_error(y_test, y_pred_multi))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_multi)))
print("R² Score:", r2_score(y_test, y_pred_multi))

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_multi, alpha=0.5, color='green')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.show()

# 12. Conclusion
print("\nConclusion:")
print("Simple Linear Regression R²:", r2_score(y_test, y_pred_simple))
print("Multiple Linear Regression R²:", r2_score(y_test, y_pred_multi))
print("Multiple regression generally performs better due to using more features.")
