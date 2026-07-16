"""
Preprocess a buoy's raw NDBC .txt files into a single cached pickle.

Only needed once per buoy — reads buoy_data/{BUOY_ID}/{density,alpha1,alpha2,
r1,wind}.txt, reindexes to a full hourly index, interpolates gaps (circular-
aware for alpha1/alpha2), and writes buoy_data/{BUOY_ID}/processed_data.pkl,
the file every other script (scripts/optimize.py, scripts/train.py,
scripts/infer.py) reads via pd.read_pickle.

Run manually:
    python scripts/data_processing.py
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path

from utils import data_processing

# Edit to the buoy you want to (re)process.
BUOY_ID = "42056"

project_root = Path(__file__).resolve().parent.parent
folder_path = project_root / "buoy_data" / BUOY_ID
save_path = folder_path / "processed_data.pkl"

data_processing(str(folder_path), save_path=str(save_path))
