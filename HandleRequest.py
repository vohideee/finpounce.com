from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    with open("preference_prediction_model.pkl", 'rb') as file:  
        model = pickle.load(file)
    label_encoder = LabelEncoder()
    categories = ["Grocery","Entertainment","Technology","Subscriptions","Clothes","Fast Food","Other"]

    data = request.get_json()
    print(data)
    input_data =  pd.DataFrame({
        'Category': data['Category'],
        'Cost': data['Cost'],
        'BalanceBefore': data['BalanceBefore'],
        'days_since_last_purchase': data['days_since_last_purchase']
    })
    print(input_data)
    input_data['Category'] = label_encoder.fit_transform(input_data['Category'])
    result = model.predict(input_data)
    label_encoder.fit(categories)
    result = label_encoder.inverse_transform(result)
    result = result.tolist()
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(port=5000)