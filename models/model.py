import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load the dataset
data = pd.read_csv('./data/car_insurance_premium_dataset.csv')

#convert categorical columns to numerical values
data['Coverage Type'] = data['Coverage Type'].map({'Third Party': 0, 'Comprehensive': 1})
data['Car Type'] = data['Car Type'].map({'suv': 0, 'pickup': 1, 'sedan': 2, 'hatchback': 3})

#choosing the features and target variable
X = data[["Driver Age", "Driver Experience", "Previous Accidents", "Annual Mileage (x1000 km)", "Car Age", "Coverage Type", "Car Type"]]
y = data["Insurance Premium ($)"]

#Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Print the results
print(f"Mean Squared Error: {mse:.2f}")
print(f'R^2 Score: {r2:.2f}')

#Example input for prediction
new_example = pd.DataFrame({
    'Driver Age': [25],
    'Driver Experience': [6],
    'Previous Accidents': [2],
    'Annual Mileage (x1000 km)': [22],
    'Car Age': [3],
    'Coverage Type': [1], #0 for third party, 1 for comprehensive 
    'Car Type': [2] #0 for suv, 1 for pickup, 2 for sedan, 3 for hatchback 
})

#Get prediction for the new example
new_prediction = model.predict(new_example)
print(f"Predicted Insurance Premium for new example: ${new_prediction[0]:.2f}")