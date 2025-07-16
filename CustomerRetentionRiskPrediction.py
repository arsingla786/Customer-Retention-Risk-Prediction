'''1. Overview
Customer churn, or customer attrition, refers to when a customer ceases their relationship
with a company or service provider. In today's highly competitive business environment,
retaining customers is a critical factor for long-term success.
Predicting customer churn, can help organisations 
take proactive steps to retain customers, thus minimizing revenue loss.
This project aims to build a machine learning model that can predict
whether a customer will churn based on their demographic, account,
and service-related data.

2. Problem Statement
The goal of this project is to develop a classification model that predicts whether 
customer
will churn. Using demographic data (such as gender, senior citizen status, and tenure),
along
with information about the services they use (such as internet service, phone service, and
online security), we will attempt to build a model that helps the company identify
customers who are at a high risk of churning.

By predicting customer churn, the company can proactively design retention strategies to
keep these customers, thereby improving customer satisfaction and reducing financial loss.
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

#load the dataset using pandas 
df = pd.read_csv("C:\\Users\\Arnav Singla\\Downloads\\Customer_data.csv")

#print some rows of data
print(df.head())

#Through graphs, analyse the data

plt.figure(figsize=(5,7))
sns.countplot(data=df , x='gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of Customers based on gender')
plt.show()

#Churn distribution
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.show()

#Monthly charges vs Churn
#Higher the monthly charges higher the chances of churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='Set3')
plt.title('Monthly Charges vs Churn')
plt.xlabel('Churn')
plt.ylabel('Monthly Charges')
plt.show()

#heatmap between all factors
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#Encode the data which is non numeric like gender , Payment methods, subscription status4

label_encoders = {}
for column in ['gender','Partner','Dependents','PhoneService','MultipleLines',
               'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
               'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling',
               'PaymentMethod','Churn']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le 

#print encoded data 
print(df.head())

#define features(factors used to predict) and target(what to predict)

#features
X = df.drop(['gender','Partner','Dependents','PhoneService','MultipleLines',
               'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
               'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling',
               'PaymentMethod'],axis=1) 

#target
y= df['Churn']

#Split into test and training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scaling - to  maintain the range like [0-1] or [0-1000] of each data features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Logistic regression model 
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
logreg_pred = logreg.predict(X_test)
logreg_accuracy = accuracy_score(y_test,logreg_pred)

#display the results 
print(f'Accuracy: {logreg_accuracy * 100:.2f}%')
print("\nClassification Report:")
print(classification_report(y_test, logreg_pred))

