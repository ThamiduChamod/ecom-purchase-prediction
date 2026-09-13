import pandas as pd

def build_customer_features(df_clean: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """
    Use only history up to cutoff_date to avoid leakage.
    """
    hist = df_clean[df_clean["InvoiceDate"] <= cutoff_date].copy()

    cust = hist.groupby("CustomerID").agg(
        total_orders=("InvoiceNo", "nunique"),
        total_spent=("TotalPrice", "sum"),
        total_items=("Quantity", "sum"),
        unique_products=("StockCode", "nunique"),
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        country=("Country", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
    ).reset_index()

    # Core engineered features
    cust["avg_order_value"] = cust["total_spent"] / cust["total_orders"]
    cust["days_since_last_purchase"] = (cutoff_date - cust["last_purchase"]).dt.days
    cust["tenure_days"] = (cutoff_date - cust["first_purchase"]).dt.days.clip(lower=1)
    cust["purchase_frequency_per_month"] = cust["total_orders"] / (cust["tenure_days"] / 30.0)

    # Window features (extra feature engineering)
    def window_agg(days: int) -> pd.DataFrame:
        start = cutoff_date - pd.Timedelta(days=days)
        w = hist[hist["InvoiceDate"] > start]
        g = w.groupby("CustomerID").agg(
            orders=("InvoiceNo", "nunique"),
            spend=("TotalPrice", "sum"),
            items=("Quantity", "sum"),
        )
        g.columns = [f"orders_last_{days}d", f"spend_last_{days}d", f"items_last_{days}d"]
        return g

    w30 = window_agg(30)
    w90 = window_agg(90)

    features = cust.set_index("CustomerID").join([w30, w90]).fillna(0).reset_index()

    # drop raw dates (we used them)
    features = features.drop(columns=["last_purchase", "first_purchase"])

    return features