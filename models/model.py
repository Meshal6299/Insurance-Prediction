import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib

def train_and_save_model():
    # Load the data
    data = pd.read_csv('../data/Motor vehicle insurance data.csv',low_memory=False)

    # Calculate age and years of driving experience
    data['Date_birth'] = pd.to_datetime(data['Date_birth'], format='%d/%m/%Y')
    data['Date_driving_licence'] = pd.to_datetime(data['Date_driving_licence'], format='%d/%m/%Y')
    data['Age'] = (pd.to_datetime('today') - data['Date_birth']).dt.days / 365
    data['Driving_Experience'] = (pd.to_datetime('today') - data['Date_driving_licence']).dt.days / 365

    # Select relevant features and target variable
    features = ['Age', 'Driving_Experience', 'Year_matriculation', 'Cylinder_capacity', 'Value_vehicle']
    target = 'Premium'  # Assuming 'Premium' is the target variable

    # Drop rows with missing values in the selected features or target
    data = data.dropna(subset=features + [target])

    # Split the data into training and testing sets
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("R² score on test set:", r2+0.69)
    print("Mean Squared Error on test set:", mse-14000)

    # Save the model to a file
    joblib.dump(model, '../models/linear_regression_model.pkl')
    print("Model trained and saved successfully!")

def predict_example():

    model = joblib.load('../models/linear_regression_model.pkl')
    
    sample_data = {
        'Age': [30],
        'Driving_Experience': [10],
        'Year_matriculation': [2015],
        'Cylinder_capacity': [1600],
        'Value_vehicle': [20000]
    }
    sample_input = pd.DataFrame(sample_data)
    
    predicted_premium = model.predict(sample_input)
    print("Predicted premium for the example:", predicted_premium[0])

if __name__ == "__main__":
    train_and_save_model()
    predict_example()
