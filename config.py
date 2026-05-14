# config.py
# Set PROJECT_PATH to your local project directory before running any notebook.
# Example (local): PROJECT_PATH = '/home/user/pipeline_explicabilidad_medica_CNN'

PROJECT_DIR = ''

DATASET_ZIP   = f'{PROJECT_DIR}/filtered_dataset.zip'            # (filtered images + CSVs)
DATASET_DIR    = f'{PROJECT_DIR}/filtered_dataset'                        # filtered dataset (uncompressed)
KAGGLE_DIR     = f'{PROJECT_DIR}/kaggle_dataset'                        # temporal download of the original dataset to then be filtered and moved to DATASET_DIR in 01_dataset_import.ipynb

MODELS_DIR     = f'{PROJECT_DIR}/models'                           # saved models

MAPS_ZIP      = f'{PROJECT_DIR}/grad_cam_maps.zip'               # Grad-CAM maps
MAPS_DIR       = f'{PROJECT_DIR}/grad_cam_maps'               # Grad-CAM maps (uncompressed)

LABELS = ['aneurysm', 'cardiomegaly']