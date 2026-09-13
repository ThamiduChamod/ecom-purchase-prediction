import pandas as pd

def make_binary_label_next_30_days(df_clean: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.Series:
    """
    Label window: (cutoff_date, cutoff_date + 30days]
    1 if customer purchased in that window else 0
    """
    label_end = cutoff_date + pd.Timedelta(days=30)

    future = df_clean[(df_clean["InvoiceDate"] > cutoff_date) & (df_clean["InvoiceDate"] <= label_end)]

    y = (future.groupby("CustomerID")["InvoiceNo"].nunique() > 0).astype(int)
    y.name = "purchase_next_30_days"
    return y