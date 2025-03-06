import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('./data/car_insurance_premium_dataset.csv')

# Manual encoding of the 'Coverage Type' column
data['Coverage Type'] = data['Coverage Type'].map({'Third Party': 0, 'Comprehensive': 1})
data['Car Type'] = data['Car Type'].map({'suv': 0, 'pickup': 1, 'sedan': 2, 'hatchback': 3})

# Include 'Car Type' in your features
X = data[["Driver Age", "Driver Experience", "Previous Accidents", "Annual Mileage (x1000 km)", "Car Age", "Coverage Type", "Car Type"]]
y = data["Insurance Premium ($)"]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error: {mse:.2f}")
print(f'R^2 Score: {r2}')
