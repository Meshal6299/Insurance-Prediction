# Car Insurance Premium Prediction Web Application

This project is a Flask-based web application that allows users to input details about a driver and their car to predict car insurance premiums. The application utilizes a machine learning model to predict the premium based on factors such as driver age, driving experience, previous accidents, car age, type of insurance, and car type. The prediction is displayed in a popup without refreshing the page, using AJAX to enhance user interaction.

## Features

- **Form Input**: Users can input details including driver age, driving experience, previous accidents, car age, insurance type, and car type.
- **Dynamic Prediction Display**: Predictions are displayed in a popup modal directly on the page without a page reload.
- **Interactive UI**: The application includes basic styling and interactive elements for a user-friendly experience.

## Technology Stack

- **Flask**: Backend framework used to create and manage the web server.
- **JavaScript**: Used for handling form submissions and displaying the popup modal dynamically.
- **HTML/CSS**: For structuring and styling the web interface.
- **Pandas**: Used for data manipulation before making predictions.
- **Scikit-learn**: For building and utilizing the linear regression model.

## Project Structure

/your-project-directory
 │ 
 ├── api/ 
 │    └── app.py # Main Flask application file
 ├── models/
 │    └── model.py # Contains the machine learning model 
 │    └── init.py
 ├── data/
 │    └── car_insurance_premium_dataset.csv # Contains the dataset needed for the model
 │    └── init.py
 ├── static/
 │    └── style.css # CSS file for styling the web interface
 ├── templates/
 │    └── index.html # Main HTML file for the web interface
 │── README.md # This file


 
## Setup and Running the Project

Follow these steps to set up and run the project on your local machine:

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-project-repository.git
   cd your-project-repository

2. **Install Required Python Libraries**:
   ```bash
   pip install flask pandas scikit-learn

3. **Start the Flask Application**:
   ```bash
   python3 -m api.app

### Prerequisites

 1. Open the application in a web browser as per the setup instructions.
 2. Fill in the form with the relevant details about the driver and car.
 3. Click 'Submit' to see the prediction of the insurance premium displayed in a popup modal.
 4. You can close the popup by clicking 'X' or outside the popup area, which will also clear the form for new entries.