from flask import Flask, request, jsonify
import joblib
import requests

app = Flask(__name__)

model = joblib.load("IoT_model.pkl")

THINGSBOARD_URL = "http://localhost:9090/api/v1/fPswlnebLOEvrRu8zSeH/IoT_sample"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    temperature = data["temperature"]
    humidity = data["humidity"] 
    PredictedSoilMoisture = data["Predicted Soil Moisture"]
    Nitrogen = data["Nitrogen"]
    Phosphorus = data["Phosphorus"]
    Potassium = data["Potassium"]

    features = [[temperature, humidity,PredictedSoilMoisture,Nitrogen,Phosphorus,Potassium]]
    prediction = model.predict(features)[0]

    payload = {
        "temperature": temperature,
        "humidity": humidity,
        "Predicted Soil Moisture":PredictedSoilMoisture,
        "Nitrogen": Nitrogen,
        "Phosphorus":Phosphorus,
        "Potassium":Potassium,
        "prediction": int(prediction)
    }

    requests.post(THINGSBOARD_URL, json=payload)

    return jsonify({
        "status": "success",
        "prediction": int(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)