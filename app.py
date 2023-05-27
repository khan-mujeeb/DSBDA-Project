
from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load the machine learning model
with open('abc.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the Flask app
app = Flask(__name__)

# Define the function to preprocess the input data
def preprocess_input(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    # Create a DataFrame with the input data
    data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })
    # Return the preprocessed data
    return data.values.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    pregnancies = int(request.form['pregnancies'])
    glucose = int(request.form['glucose'])
    blood_pressure = int(request.form['blood_pressure'])
    skin_thickness = int(request.form['skin_thickness'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
    age = int(request.form['age'])

    # Preprocess the input data
    data = preprocess_input(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)

    # Use the model to make a prediction
    prediction = model.predict(data)[0]
    print("mujeeb ")
    print(prediction)

    # Return the prediction to the user
    return render_template('output.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)


