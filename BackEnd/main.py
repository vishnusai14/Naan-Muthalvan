from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pickle

with open("../wine_model.pkl", 'rb') as file:
    wine_model = pickle.load(file)

app = Flask(__name__)

CORS(app, support_credentials=True)


@app.route("/api/v1", methods=['GET'])
@cross_origin(supports_credentials=True)
def home():
    return jsonify({'result': 'App is Running'}), 201


@app.route("/api/v1/predict", methods=['POST'])
@cross_origin(supports_credentials=True)
def predict():
    print(request)
    if request.method == 'POST':
        data = request.json
        feature = [[float(data['fA']), float(data['vA']), float(data['cA']), float(data['rS']), float(data['cL']), float(data['sO']), float(data['den']), float(data['pH']), float(data['sU']), float(data['aL'])]]
        print(feature)
        output = wine_model.predict(feature)
        response = output.tolist()
        return jsonify({'result': response}), 201


if __name__ == "__main__":
    app.run(host='192.168.190.208', port=3000, debug=True)
