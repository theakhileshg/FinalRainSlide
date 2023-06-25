from flask import Flask, render_template, request
import os
import pickle
import pandas as pd

app = Flask(__name__)

# Define the path to the pickle file
model_path = os.path.join('model', 'randomforest_model.pkl')

# Load the Random Forest model from the pickle file
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process_data():
    rain = int(request.form.get('rain'))
    slope = float(request.form.get('slope'))
    precipitation = float(request.form.get('precipitation'))
    temperature = float(request.form.get('temperature'))
    humidity = float(request.form.get('humidity'))
    speed = float(request.form.get('speed'))
    air = float(request.form.get('air'))

    input_data = pd.DataFrame({
        'trigger' : [rain],
        'precip': [precipitation],
        'temp': [temperature],
        'air': [air],
        'humidity': [humidity],
        'wind': [speed],
        'slope' : [slope],
    })
    result = model.predict(input_data)
    print(result)
    return render_template('result.html', result=result[0])

if __name__ == '__main__':
    app.run(debug=True)
