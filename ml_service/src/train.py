from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data_load import load_raw_csv
from src.clean import clean_retail_data
from src.features import build_customer_features
from src.make_labels import make_binary_label_next_30_days

ROOT = Path(__file__).resolve().parents[2]  # project root
DATA_PATH = ROOT / "data" / "raw" / "online_retail.csv"

ARTIFACTS_DIR = ROOT / "ml-service" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def pick_cutoff_date(df_clean: pd.DataFrame) -> pd.Timestamp:
    # max_date - 60d gives: history + 30d label window, and leaves tail for evaluation realism
    max_date = df_clean["InvoiceDate"].max()
    return max_date - pd.Timedelta(days=60)

def train_and_save():
    raw = load_raw_csv(DATA_PATH)
    df = clean_retail_data(raw)

    cutoff = pick_cutoff_date(df)
    X = build_customer_features(df, cutoff_date=cutoff)
    y = make_binary_label_next_30_days(df, cutoff_date=cutoff)

    data = X.merge(y.reset_index(), on="CustomerID", how="left")
    data["purchase_next_30_days"] = data["purchase_next_30_days"].fillna(0).astype(int)

    # Save feature table for API usage (CustomerID lookup)
    feature_table_path = ARTIFACTS_DIR / "customer_features.parquet"
    data.to_parquet(feature_table_path, index=False)

    target = "purchase_next_30_days"
    id_col = "CustomerID"

    X_all = data.drop(columns=[target])
    y_all = data[target]

    # define columns
    cat_cols = ["country"]
    num_cols = [c for c in X_all.columns if c not in [id_col] + cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
    }

    # split (random split is ok here because label is already time-locked via cutoff)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all.drop(columns=[id_col]),
        y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )

    best_name, best_pipe, best_auc = None, None, -1

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, pred)

        print(f"\nModel: {name}")
        print(f"ROC-AUC: {auc:.4f} | F1: {f1:.4f}")
        print(classification_report(y_test, pred))

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipe = pipe

    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(
        {
            "model_name": best_name,
            "cutoff_date": str(cutoff),
            "pipeline": best_pipe,
            "id_col": id_col,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
        },
        model_path
    )
    print(f"\nSaved model: {model_path}")
    print(f"Saved features: {feature_table_path}")
    print(f"Best model: {best_name} (ROC-AUC={best_auc:.4f})")

if __name__ == "__main__":
    train_and_save()