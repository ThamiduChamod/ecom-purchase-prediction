from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from api.schemas import PredictRequest, PredictResponse

ROOT = Path(__file__).resolve().parents[2]  # project root
ARTIFACTS_DIR = ROOT / "ml-service" / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "customer_features.parquet"

app = FastAPI(title="Customer Purchase Prediction ML Service")

bundle = None
features_df = None

@app.on_event("startup")
def load_artifacts():
    global bundle, features_df
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}. Run training first.")
    if not FEATURES_PATH.exists():
        raise RuntimeError(f"Features table not found: {FEATURES_PATH}. Run training first.")

    bundle = joblib.load(MODEL_PATH)
    features_df = pd.read_parquet(FEATURES_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/customers")
def list_customers(limit: int = Query(50, ge=1, le=500)):
    # return first N customer IDs (you can add search/pagination later)
    ids = features_df["CustomerID"].astype(str).dropna().unique().tolist()
    return {"count": len(ids), "customers": ids[:limit]}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    customer_id = str(req.customerId)

    row = features_df[features_df["CustomerID"].astype(str) == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"CustomerID not found: {customer_id}")

    # model expects same raw feature columns excluding target & id
    target = "purchase_next_30_days"
    X = row.drop(columns=[target, "CustomerID"], errors="ignore")

    pipe = bundle["pipeline"]
    proba = float(pipe.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)

    return PredictResponse(
        customerId=customer_id,
        prediction=pred,
        probability=proba,
        modelName=bundle["model_name"],
    )