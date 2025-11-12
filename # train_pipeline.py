# train_pipeline.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42

def generate_synthetic_dataset(n=20000, imbalance_ratio=0.02):
    """
    Generate synthetic transactional dataset.
    imbalance_ratio = fraction of suspicious transactions (true positives)
    """
    np.random.seed(RANDOM_STATE)
    # Basic features inspired by the doc
    transaction_amount = np.random.exponential(scale=2000, size=n)  # long-tail
    transaction_frequency = np.random.poisson(lam=3, size=n)
    offshore_flag = np.random.binomial(1, 0.05, size=n)  # small fraction offshore
    client_age = np.random.randint(18, 85, size=n)
    country_risk_score = np.random.beta(a=1.2, b=6, size=n)  # skewed low
    account_age_days = np.random.exponential(scale=365*2, size=n)
    num_payees = np.random.poisson(lam=1.2, size=n)
    avg_txn_amt_30d = np.random.exponential(scale=1500, size=n)

    # construct dataframe
    df = pd.DataFrame({
        "transaction_amount": transaction_amount,
        "transaction_frequency": transaction_frequency,
        "offshore_flag": offshore_flag,
        "client_age": client_age,
        "country_risk_score": country_risk_score,
        "account_age_days": account_age_days,
        "num_payees": num_payees,
        "avg_txn_amt_30d": avg_txn_amt_30d
    })

    # label generation heuristic: higher amount + high country risk + offshore + many payees -> more likely suspicious
    score = (
        0.0003 * df.transaction_amount +
        1.6 * df.offshore_flag +
        2.5 * df.country_risk_score +
        0.2 * (df.num_payees > 3).astype(int) +
        0.0001 * df.avg_txn_amt_30d
    )

    # convert score to probabilities with sigmoid and adjust to desired imbalance
    probs = 1 / (1 + np.exp(- (score - np.percentile(score, 95) * 0.6)))
    # scale probs so that mean approx imbalance_ratio
    probs = probs * (imbalance_ratio / probs.mean())

    y = np.random.binomial(1, np.clip(probs, 0, 1))
    df["is_suspicious"] = y
    # ensure at least a few suspicious
    if df.is_suspicious.sum() == 0:
        df.loc[df.sample(10, random_state=RANDOM_STATE).index, "is_suspicious"] = 1

    return df

def preprocess(df, scaler=None, fit_scaler=True):
    X = df.drop(columns=["is_suspicious"])
    y = df["is_suspicious"].values
    # handle simple transformations
    X["transaction_amount_log"] = np.log1p(X["transaction_amount"])
    X["avg_txn_amt_30d_log"] = np.log1p(X["avg_txn_amt_30d"])
    X = X.drop(columns=["transaction_amount","avg_txn_amt_30d"])

    numeric_cols = X.columns.tolist()
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[numeric_cols])
    else:
        X_scaled = scaler.transform(X[numeric_cols])

    X_scaled = pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)
    return X_scaled, y, scaler

def train_and_evaluate(df):
    print("Dataset shape:", df.shape, "Suspicious count:", df.is_suspicious.sum())
    X, y, scaler = preprocess(df, fit_scaler=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    # simple model definitions
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
    xgb = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)

    models = {"RandomForest": rf, "XGBoost": xgb, "SVM": svm}
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs) if probs is not None and len(np.unique(y_test))>1 else None
        prec, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        print(f"{name} -> acc: {acc:.4f}, prec: {prec:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, auc: {auc}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        results[name] = {"model": model, "acc": acc, "auc": auc, "precision": prec, "recall": recall, "f1": f1}

        # save model
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        print(f"Saved {name} to {MODEL_DIR}/{name}.joblib")

    # save scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print("Saved scaler.")

    # Feature importances for RandomForest/XGBoost
    best_rf = models["RandomForest"]
    importances = best_rf.feature_importances_
    feature_names = X.columns
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    plt.figure(figsize=(8,6))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"))
    print("Saved feature importance plot.")

    # summary
    summary = pd.DataFrame([{ "model": k, **v } for k,v in results.items()])
    summary_path = os.path.join(MODEL_DIR, "training_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved training summary to", summary_path)

    return results

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset(n=20000, imbalance_ratio=0.02)
    results = train_and_evaluate(df)
    print("Done.")
