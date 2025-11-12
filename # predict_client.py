# predict_client.py
import joblib
import numpy as np
import pandas as pd
import os

MODEL_DIR = "models"

def load_model(name="RandomForest"):
    model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError("Model or scaler not found. Run train_pipeline.py first.")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_single(record: dict, scaler):
    """
    record: dict with keys:
    transaction_amount, transaction_frequency, offshore_flag, client_age,
    country_risk_score, account_age_days, num_payees, avg_txn_amt_30d
    """
    df = pd.DataFrame([record])
    df["transaction_amount_log"] = np.log1p(df["transaction_amount"])
    df["avg_txn_amt_30d_log"] = np.log1p(df["avg_txn_amt_30d"])
    df = df.drop(columns=["transaction_amount","avg_txn_amt_30d"])
    cols = df.columns.tolist()
    X_scaled = scaler.transform(df[cols])
    X_scaled = pd.DataFrame(X_scaled, columns=cols)
    return X_scaled

def predict_single(record: dict, model_name="RandomForest"):
    model, scaler = load_model(model_name)
    X = preprocess_single(record, scaler)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0,1] if hasattr(model, "predict_proba") else None
    return {"is_suspicious": int(pred), "probability": float(prob) if prob is not None else None}

# Example usage:
if __name__ == "__main__":
    example = {
        "transaction_amount": 12000,
        "transaction_frequency": 1,
        "offshore_flag": 1,
        "client_age": 45,
        "country_risk_score": 0.8,
        "account_age_days": 150,
        "num_payees": 6,
        "avg_txn_amt_30d": 3000
    }
    print(predict_single(example, model_name="RandomForest"))
