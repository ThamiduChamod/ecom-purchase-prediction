from pathlib import Path
import pandas as pd

def load_raw_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    # Online retail datasets often require ISO-8859-1
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    return df