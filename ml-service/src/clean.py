import pandas as pd

RENAME_MAP = {
    "Invoice": "InvoiceNo",
    "Customer ID": "CustomerID",
    "Price": "UnitPrice",
}

def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # rename columns (handle if already renamed)
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    # basic required columns
    required = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # type conversions
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    # drop invalid / missing essentials
    df = df.dropna(subset=["CustomerID", "InvoiceDate", "InvoiceNo", "StockCode", "Country", "Quantity", "UnitPrice"])

    # tidy CustomerID (avoid 12345.0)
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    # remove cancellations if InvoiceNo starts with 'C' (common in this dataset)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # remove invalid quantity/price
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # line total
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # drop duplicates
    df = df.drop_duplicates()

    return df