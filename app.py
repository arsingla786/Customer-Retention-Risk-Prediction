from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

# Load the model and preprocessing objects
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')

app = Flask(__name__)
print("Flask app started successfully")

# Define input fields in the same order as training
input_fields = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            user_input = []
            
            # Collect form data
            for field in input_fields:
                value = request.form.get(field)
                try:
                    # Convert numeric fields
                    if field in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
                        val = float(value)
                        if val.is_integer():
                            val = int(val)
                    else:
                        val = value
                except ValueError:
                    val = value
                user_input.append(val)
            
            # Convert inputs into DataFrame
            df_input = pd.DataFrame([user_input], columns=input_fields)
            
            # Encode categorical fields using saved encoders
            categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                              'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                              'PaperlessBilling', 'PaymentMethod']
            
            for column in categorical_cols:
                if column in df_input.columns and column in label_encoders:
                    df_input[column] = label_encoders[column].transform([df_input[column].iloc[0]])
            
            # Scale the input data
            input_scaled = scaler.transform(df_input)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Generate result message
            if prediction == 1:
                result = "⚠️ Customer is more likely to cease the relationship with company"
                recommendations = [
                    "Identify churn drivers through usage patterns and complaints",
                    "Engage with personalized customer success representative",
                    "Offer targeted incentives like discounts or upgrades",
                    "Improve customer experience with better plans",
                    "Monitor churn signals continuously"
                ]
            else:
                result = "✅ Customer seems to be loyal"
                recommendations = [
                    "Continue tracking customer behavior",
                    "Maintain high service quality",
                    "Consider loyalty rewards program",
                    "Regular check-ins to ensure satisfaction"
                ]
            
            return render_template('index.html', 
                                 result=result, 
                                 recommendations=recommendations,
                                 prediction=prediction,
                                 values=request.form)
        
        except Exception as e:
            error_msg = f"Error processing prediction: {str(e)}"
            return render_template('index.html', error=error_msg)
    
    # GET request - show empty form
    return render_template('index.html', result=None)
print('no problem in lfask')
if __name__ == '__main__':
    app.run(debug=True)


