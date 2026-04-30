from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "ihdp_data.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"

RANDOM_STATE = 42
TEST_SIZE = 0.3

MLFLOW_EXPERIMENT_NAME = "DML_Causal_Inference_v5"