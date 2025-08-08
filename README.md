# Customer Loyalty Prediction 
 
This project is a machine learning-powered web application that predicts whether a customer is likely to churn based on their behavioral and demographic attributes. The goal is to assist businesses in identifying high-risk customers and taking proactive steps to retain them.

## 🚀 Web Application
[Visit Application](https://customer-loyalty-prediction.up.railway.app/)  


---
## 📂 Project Structure

```

model/
├── model.pkl                  # Trained ML model (Random Forest)
├── scaler.pkl                 # Feature scaler (StandardScaler)
├── label_encoders.pkl         # Label encoders for categorical features
templates/
├── index.html                 # Frontend template for input form
app.py                         # Flask web app
Customer_data.csv              # Dataset used for training
Customer_Loyalty_Prediction.py # Training + model saving script
Procfile                       # Deployment process file (for Railway/Heroku)
requirements.txt               # Python dependencies
```
---

## 🧠 Model

- **Algorithms Used:** Random Forest Classifier , Logistic regression
- **Libraries:** scikit-learn, pandas, numpy, joblib, matplotlib, seaborn
- **Features:** Customer demographics, tenure, charges, service usage, etc.
- **Target:** Predict whether a customer will **churn** or **stay loyal**

---

## 🖥️ Web App Features

- Input customer data via a form
- Predict customer loyalty instantly
- Simple and clean UI built with Flask and HTML
- Get suggestions based on the prediction

<br><br>
___
<br><br>

<img width="746" height="500" alt="Screenshot 2025-07-19 141223" src="https://github.com/user-attachments/assets/b3831052-cbc4-4052-8795-3cfe81c57e53" />
