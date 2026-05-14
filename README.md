# chest_xray_xai_pipeline
Training, evaluation and explicability pipeline with Grad-CAM for CNN models that classify radiological images, with a focus on the clinical validation of the models' activations.

# General pre requisites
--Complete config.py: add the path where the project will generate

# Pre requisites for 01_import_dataset
--Complete verification course from https://physionet.org/content/vindr-cxr/1.0.0/
--Have a kaggle account
---Apply for this contest https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection

# Pre requisites for 02_preprocessing_images_and_training
--Have a dataset inside PROJECT_PATH/dataset with two resolutions -> PROJECT_PATH/dataset/images_256 and PROJECT_PATH/dataset/images_1024
