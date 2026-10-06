"""Read the JADER CSV files (Shift-JIS/CP932) and save them as parquet in WORK_DIR."""
import os
import pandas as pd
from common import WORK_DIR, jader_file

for kind in ["demo", "drug", "reac", "hist"]:
    df = pd.read_csv(jader_file(kind), dtype=str, encoding="cp932", encoding_errors="replace")
    df.to_parquet(os.path.join(WORK_DIR, f"jader_{kind}.parquet"))
    print(kind, len(df))
