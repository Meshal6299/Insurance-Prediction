from flask import Flask, render_template, request, jsonify
from models.model import model
import pandas as pd

app = Flask(__name__, template_folder='../templates', static_folder='../static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get data from form
    data = {
        'Driver Age': [request.form.get('driverAge', type=int)],
        'Driver Experience': [request.form.get('driverExperience', type=int)],
        'Previous Accidents': [request.form.get('previousAccidents', type=int)],
        'Car Age': [request.form.get('carAge', type=int)],
        'Coverage Type': [request.form.get('insurance')],  # Ensure categorical encoding if needed
        'Car Type': [request.form.get('carType')]  # Ensure categorical encoding if needed
    }

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame(data)

    # Assume 'predict' method expects a DataFrame
    prediction = model.predict(input_df)
    
    formatted_prediction = f"{prediction[0]:.2f}"

    # Return a JSON response with the Predicion
    return jsonify({'prediction': formatted_prediction})

if __name__ == '__main__':
    app.run(debug=True)
