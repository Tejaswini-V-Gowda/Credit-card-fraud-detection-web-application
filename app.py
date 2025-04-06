from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join('model', 'fraud_detection_model.pkl')
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the form
        input_str = request.form['input1']
        input_values = [float(x) for x in input_str.split(',')]
        input_array = np.array([input_values])

        # Predict
        prediction = model.predict(input_array)[0]

        return render_template('index.html', prediction=int(prediction))
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
