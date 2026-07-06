from pathlib import Path

# Project
project_root = Path(__file__).resolve().parent.parent
figures_path = project_root/"figures"
results_path = project_root/"results"

# Dataset
dataset_root = Path("C:/MyProjects/MIMIIResearch/id_00")
normal_path = dataset_root/"normal"
abnormal_path = dataset_root/"abnormal"

# Experiment
machine = "Pump"
machine_id = "id_00"
channel = 0
random_seed = 42