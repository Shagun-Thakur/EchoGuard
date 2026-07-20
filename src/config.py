from pathlib import Path

# Project
project_root = Path(__file__).resolve().parent.parent
results_path = project_root/"results"

# Dataset
app_dataset_root = project_root/"sample_dataset_for_app"

try:
    from src.local_config import TRAIN_DATASET_PATH
except ImportError:
    TRAIN_DATASET_PATH = None

def get_training_dataset_path():
    """
    Returns the normal and abnormal paths of the full training dataset.
    """
    if TRAIN_DATASET_PATH is None:
        raise FileNotFoundError(
            "TRAIN_DATASET_PATH is not configured.\n"
            "Create src/local_config.py and define TRAIN_DATASET_PATH"
        )
    return{
        "normal" : TRAIN_DATASET_PATH / "normal",
        "abnormal" : TRAIN_DATASET_PATH / "abnormal"
    }

def get_app_dataset_paths():
    """
    Returns the normal and abnormal paths of the sample dataset
    used by the Streamlit application.
    """
    return{
        "normal": app_dataset_root / "normal",
        "abnormal": app_dataset_root/ "abnormal"
    }

# Experiment Configuration

machine = "Pump"
machine_id = "id_00"
channel = 0
random_seed = 42