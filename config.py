# config.py
# Set PROJECT_PATH to your local project directory before running any notebook.
# Example (local): PROJECT_PATH = '/home/user/pipeline_explicabilidad_medica_CNN'

PROJECT_PATH = ''

DATASET_DIR   = f'{PROJECT_PATH}/filtered_dataset'    # (filtered images + CSVs)
KAGGLE_DIR      = f'{PROJECT_PATH}/dataset'           # temporal download of the original dataset to then be filtered and moved to DATASET_DIR in 01_dataset_import.ipynb
MODELS_DIR    = f'{PROJECT_PATH}/models'              # saved models
MAPS_DIR      = f'{PROJECT_PATH}/maps'                # Grad-CAM maps