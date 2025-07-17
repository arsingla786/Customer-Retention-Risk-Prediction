from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import traceback

# Check current working directory
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# Create Flask app with explicit template folder
app = Flask(__name__, template_folder='templates')

# Debug: Check if files exist
print("Checking files...")
print(f"Model file exists: {os.path.exists('model/model.pkl')}")
print(f"Scaler file exists: {os.path.exists('model/scaler.pkl')}")
print(f"Label encoders file exists: {os.path.exists('model/label_encoders.pkl')}")
print(f"Template folder exists: {os.path.exists('templates')}")
print(f"Template index.html exists: {os.path.exists('templates/index.html')}")
print(f"Template test.html exists: {os.path.exists('templates/test.html')}")

if os.path.exists('templates'):
    print(f"Files in templates folder: {os.listdir('templates')}")

try:
    # Load the model and preprocessing objects
    print("Loading models...")
    model = joblib.load('model/model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    traceback.print_exc()

@app.route('/test')
def test():
    return "<h1>App is running!</h1><p>Flask routing works fine.</p>"

@app.route('/simple')
def simple():
    try:
        print("Attempting to render test.html...")
        return render_template('test.html')
    except Exception as e:
        print(f"Error rendering simple template: {e}")
        traceback.print_exc()
        return f"<h1>Template Error</h1><p>Error: {str(e)}</p><p>Check if templates/test.html exists</p>"

@app.route('/raw')
def raw():
    # Try to read the template file directly
    try:
        with open('templates/test.html', 'r') as f:
            content = f.read()
        return f"<h1>Raw Template Content:</h1><pre>{content}</pre>"
    except Exception as e:
        return f"<h1>Cannot read template file</h1><p>Error: {str(e)}</p>"

@app.route('/')
def home():
    try:
        print("Attempting to render index.html...")
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering index template: {e}")
        traceback.print_exc()
        return f"<h1>Index Template Error</h1><p>Error: {str(e)}</p>"

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)   