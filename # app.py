# app.py
from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import subprocess
from predict_client import predict_single, load_model
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST","GET"])
def train():
    # run training script (for demo; in prod use jobs/async worker)
    # Here we call the training script synchronously (quick for small data)
    output = subprocess.run(["python", "train_pipeline.py"], capture_output=True, text=True)
    return "<pre>" + output.stdout + "\n\n" + output.stderr + "</pre>"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()
    # convert fields to correct types
    def parse(k):
        val = data.get(k)
        if val is None:
            return 0
        try:
            if "." in val:
                return float(val)
            return int(val)
        except:
            try:
                return float(val)
            except:
                return val

    record = {
        "transaction_amount": float(data.get("transaction_amount", 0)),
        "transaction_frequency": int(data.get("transaction_frequency", 0)),
        "offshore_flag": int(data.get("offshore_flag", 0)),
        "client_age": int(data.get("client_age", 30)),
        "country_risk_score": float(data.get("country_risk_score", 0.1)),
        "account_age_days": float(data.get("account_age_days", 365)),
        "num_payees": int(data.get("num_payees", 0)),
        "avg_txn_amt_30d": float(data.get("avg_txn_amt_30d", 0))
    }

    model_name = data.get("model_name", "RandomForest")
    try:
        result = predict_single(record, model_name=model_name)
        return render_template("result.html", record=record, result=result, model_name=model_name)
    except Exception as e:
        return f"Error: {e}", 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json()
    model_name = payload.get("model_name", "RandomForest")
    record = payload.get("record")
    if record is None:
        return jsonify({"error": "provide 'record' in json payload"}), 400
    try:
        result = predict_single(record, model_name=model_name)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
