import numpy as np
import joblib

from flask import Flask, request, jsonify

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


#with open('model1.bin', 'rb') as f_in:
#    dv, model = pickle.load(f_in)

model = joblib.load('model1.bin')
dv = joblib.load('dv.bin')


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    give_credit = prediction >= 0.5
    
    result = {
        'credit_probability': float(prediction),
        'can_have_credit': bool(give_credit),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)