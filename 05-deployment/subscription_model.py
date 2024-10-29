import pickle
import numpy as np

from flask import Flask, request, jsonify

app = Flask('subscription')

with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

def predict_single(client, dv, model):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    prediction = predict_single(client, dv, model)
    subscription = prediction >= 0.5
    
    result = {
        'subscription_probability': float(prediction),
        'subscription': bool(subscription),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)