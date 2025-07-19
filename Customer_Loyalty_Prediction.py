'''1. Overview
Customer churn, or customer attrition, refers to when a customer ceases their relationship
with a company or service provider. In today's highly competitive business environment,
retaining customers is a critical factor for long-term success.
Predicting customer churn, can help organisations 
take proactive steps to retain customers, thus minimizing revenue loss.
This project aims to build a machine learning model that can predict
whether a customer will churn based on their demographic, account,
and service-related data.'''

import pandas as pd
print(1)   

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE

#load the dataset using pandas 
df = pd.read_csv("C:\\Users\\Arnav Singla\\Downloads\\Customer_data.csv")

# Convert TotalCharges to numeric and fill missing with 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Drop customerID
df.drop(columns=['customerID'], inplace=True)

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
X = df.drop('Churn',axis=1) 

#target
y= df['Churn']

# Handle class imbalance using SMOTE (if in data one category is way more than other)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#Split into test and training data

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

# Scaling - to  maintain the range like [0-1] or [0-1000] of each data features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#______________Logistic regression model_____________
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_train,y_train)
logreg_pred = logreg.predict(X_test)
logreg_accuracy = accuracy_score(y_test,logreg_pred)

#display the results 
print('FOR LOGISTIC REGRESSION MODEL : ')
print(f'Accuracy: {logreg_accuracy * 100:.2f}%')
print("\nClassification Report:")
print(classification_report(y_test, logreg_pred))

print('end1')
#______________RANDOM FOREST MODEL___________________
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train,y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test,rf_pred)

#display the results 
print('FOR RANDOM FOREST MODEL : ')
print(f'Accuracy: {rf_accuracy * 100:.2f}%')
print("\nClassification report :")
print(classification_report(y_test,rf_pred))

print("Random Forest model is better")

#plotting the accuracies of both models
accuracies = [logreg_accuracy * 100, rf_accuracy * 100]
model_names = ['Logistic Regression', 'Random Forest']

# Plot
plt.figure(figsize=(6, 4))
bars = plt.bar(model_names, accuracies, color=['skyblue', 'orange'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.show()

#Taking a sample input from user 

'''input_fields = [
    'gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService',
    'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
    'Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'
]


user_values = []
print("Enter the following values as prompted (0/1 for binary, or numeric values):")
for col in input_fields:
    val = input(f"{col}: ")
    try:
        val = float(val)
        if val.is_integer():
            val = int(val)
    except ValueError:
        pass
    user_values.append(val)

# Convert to DataFrame
input_df = pd.DataFrame([user_values], columns=input_fields)

# Encode categorical fields using saved LabelEncoders
categorical_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines',
                    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies','Contract',
                    'PaperlessBilling','PaymentMethod']

for col in categorical_cols:
    le = label_encoders[col]
    input_df[col] = le.transform([input_df[col][0]])

# Reorder columns as per training features
final_columns = X.columns  # X is training feature set
input_df = input_df[final_columns]

# Scale
input_scaled = scaler.transform(input_df)

# Convert to NumPy array
input_array = np.array(input_scaled)

# Predict
prediction = rf_model.predict(input_array)[0]'''

# Result
prediction=1
print("\nChurn Prediction:")
print("Customer is more likely to cease the relationship with company" if prediction == 1 else "Customer seems to be loyal")

if prediction==1:
    print('''Here are some precautions to take: 
 1. Identify Churn Drivers
Analyze the customer's usage patterns, complaints, or drop in engagement.

Check for:

Frequent service issues

High monthly charges

Low tenure or recent plan downgrade

2. Engage with the Customer
Assign a customer success representative to personally reach out.

Use surveys or calls to understand dissatisfaction.

Acknowledge their concerns — prevention starts with communication.

 3. Offer Personalized Incentives
Discounts, cashback, or loyalty points

Free upgrades (e.g., faster internet, premium features)

Limited-time offers tailored to their usage

 4. Improve Their Experience
Suggest better or more relevant plans based on their needs

Offer faster support or assign a dedicated service manager

Remove or simplify hidden charges or billing confusion

5. Monitor Churn Signals Continuously
Automate churn alerts for high-risk customers (via your model)

Track:

Service usage decline

Increasing complaints

Payment delays

 6. Build Customer Trust
Send regular updates about new features or policies

Be transparent in pricing, downtime, and service updates

Ensure customer data privacy and respectful communication'''

    )

else:
    print('Keep tracking the customer behaviour!')

import os
import joblib

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

joblib.dump(rf_model,'model/model.pkl')
joblib.dump(scaler,'model/scaler.pkl')
joblib.dump(label_encoders,'model/label_encoders.pkl')

