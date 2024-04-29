import pickle
import numpy as np
import pandas as ps
import sklearn
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, flash
import logging

app = Flask(__name__ ,template_folder="templates")
app.config['SECRET_KEY'] = 'mysecretkey'

# Set up logging
logging.basicConfig(level=logging.DEBUG)

model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
data = ps.read_csv("car_data.csv")

def extract_features(request_data):
    Year = int(request_data['year_name'])
    Present_Price = float(request_data['Present_Price'])
    Kms_Driven = int(request_data['Kms_Driven'])
    Owner = int(request_data['owner'])
    Fuel_Type_Petrol = request_data['Fuel_Type_Petrol']
    if Fuel_Type_Petrol == 'Petrol':
        Fuel_Type_Petrol = 1
        Fuel_Type_Diesel = 0
    elif Fuel_Type_Petrol == 'Diesel':
        Fuel_Type_Petrol = 0
        Fuel_Type_Diesel = 1
    else:
        Fuel_Type_Petrol = 0
        Fuel_Type_Diesel = 0
    Year = 2020 - Year
    Seller_Type_Individual = request_data['Seller_Type_Individual']
    if Seller_Type_Individual == 'Individual':
        Seller_Type_Individual = 1
    else:
        Seller_Type_Individual = 0
    Transmission_Mannual = request_data['Transmission_Mannual']
    if Transmission_Mannual == 'Mannual':
        Transmission_Mannual = 1
    else:
        Transmission_Mannual = 0

    feature_data = data[
        (data['Year'] == Year) &
        (data['Present_Price'] == Present_Price) &
        (data['Kms_Driven'] == Kms_Driven) &
        (data['Owner'] == Owner) &
        (data['Fuel_Type_Petrol'] == Fuel_Type_Petrol) &
        (data['Seller_Type_Individual'] == Seller_Type_Individual) &
        (data['Transmission'] == Transmission_Mannual)
    ]

    X = feature_data[['Present_Price', 'Kms_Driven', 'Owner', 'Year', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Seller_Type_Individual', 'Transmission_Mannual']].values

    X = standard_to.transform(X)

    return X

standard_to = StandardScaler()

@app.route('/')
def home():
    model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
    data = ps.read_csv("car_data.csv")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            X = extract_features(request.form)
            prediction = model.predict(X)
            output = round(prediction[0], 2)
            if output < 0:
                flash('Sorry, you cannot sell this car.')
                return render_template('index.html', prediction_text="Sorry, you cannot sell this car.")
            else:
                return render_template('index.html', prediction_text='Predicted selling price: {} lakhs.'.format(output))
    except Exception as e:
        app.logger.error(e)
        flash('An error occurred while processing your request.')
    return render_template('index.html', prediction_text='14.02')



if __name__=="__main__":
    app.run()